//
// Created by alex2772 on 2/27/26.
//

#include "OpenAIChatImpl.h"

#include <chrono>
#include <optional>
#include <random>
#include <range/v3/algorithm/generate.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view.hpp>

#include "AUI/Curl/ACurl.h"
#include "AUI/Curl/AFormMultipart.h"
#include "AUI/IO/AFileOutputStream.h"
#include "AUI/Image/jpg/JpgImageLoader.h"
#include "AUI/Json/Conversion.h"
#include "AUI/Logging/ALogger.h"
#include "AUI/Reflect/AEnumerate.h"
#include "AUI/Util/kAUI.h"
#include "config.h"
#include "AUI/IO/AByteBufferInputStream.h"
#include "AUI/Util/ATokenizer.h"
#include "util/openai_streaming.h"
#include "util/tui_streaming.h"

static constexpr auto LOG_TAG = "OpenAIChat";

using namespace std::chrono_literals;



AString IOpenAIChat::embedImage(AImageView image) {
    ALOG_TRACE(LOG_TAG) << "embedImage";
    AByteBuffer jpg;
    auto resized = image.resizedLinearDownscale({672, 672});
    JpgImageLoader::save(jpg, resized);
    // JpgImageLoader::save(AFileOutputStream("test.jpg"), resized);
    return "<{}>data:image/jpg;base64,{}</{}>"_format(EMBEDDING_TAG, jpg.toBase64String(), EMBEDDING_TAG);
}


AJson OpenAIChatImpl::makeQueryString(Params params, const IOpenAIChat::Session& messages) {
    ALOG_TRACE(LOG_TAG) << "makeQueryString";
    AUI_ASSERT(!messages.sessionId.empty());
    AJson json {
        {
          "messages",
          aui::to_json(messages),
        },
        { "max_tokens", params.maxOutputTokens },   // hopefully helps with stuck prediction (infinite reasoning)
        { "stream", false },
        { "model", params.config.model },
        { "tools", params.tools },
    };

    if (config().llmTemperature) {
        json["temperature"] = *config().llmTemperature;
    }
    if (config().llmTopP) {
        json["top_p"] = *config().llmTopP;
    }
    if (config().llmTopK) {
        json["top_k"] = *config().llmTopK;
    }
    if (config().llmMinP) {
        json["min_p"] = *config().llmMinP;
    }
    if (config().llmPresencePenalty) {
        json["presence_penalty"] = *config().llmPresencePenalty;
    }
    if (config().llmRepetitionPenalty) {
        json["repeat_penalty"] = *config().llmRepetitionPenalty;
    }
    if (params.seed) {
        json["seed"] = *params.seed;
    }
    return json;
}

AFuture<AJson> OpenAIChatImpl::makeHttpRequest(Endpoint endpoint, std::string query, std::string_view sessionId) {
    ALOG_TRACE(LOG_TAG) << "Query: " << query;
    AVector<AString> headers = {"Content-Type: application/json", "x-session-id: {}"_format(sessionId) };
    if (!endpoint.bearerKey.empty()) {
        headers << "Authorization: Bearer {}"_format(endpoint.bearerKey);
    }

    tryAgain:
    auto response = AJson::fromBuffer((co_await ACurl::Builder(endpoint.baseUrl + "chat/completions")
                                           .withMethod(ACurl::Method::HTTP_POST)
                                           .withTimeout(config().requestTimeoutSecs)
                                           .withHeaders(headers)
                                           .withBody(query)
                                           .runAsync())
                                          .body);

    if (response.contains("error")) {
        auto message = AJson::toString(response["error"]);
        if (message.contains("model failed to load, this may be due to resource limitations or an internal error")) {
            // if vram is VERY low, ollama even fails to unload the previous model.
            // in ollama_setup.sh, we set OLLAMA_KEEP_ALIVE=1m, so we will just wait the ollama to unload the model,
            // then try again.
            ALogger::warn(LOG_TAG) << "Ollama model failed to load, wait and retry...";
            co_await AThread::asyncSleep(1min / 2);
            goto tryAgain;
        }
        throw AException("Ollama error: " + message);
    }
    co_return response;
}


_<IOpenAIChat::StreamingResponse> OpenAIChatImpl::chatStreaming(Params params, IOpenAIChat::Session messages) {
    messages.insert(messages.begin(), {Message::Role::SYSTEM_PROMPT, params.systemPrompt});
    AString query = [&] {
        auto json = makeQueryString(params, messages);
        json["stream"] = true;
        // Without this, most OpenAI-compatible SSE streams (ollama.com included) never emit a final
        // usage-only chunk, so Response::usage silently stays zeroed for the entire request. That in turn
        // means AppBase's diary-dump trigger (mTemporaryContext usage.total_tokens >= diaryTokenCountTrigger)
        // can never fire for a streaming provider that needs this flag - the diary would never get written to,
        // no matter how long the conversation runs.
        json["stream_options"] = AJson::Object{{"include_usage", true}};
        return AJson::toString(json);
    }();
    AFileOutputStream("last_query.json") << query.toStdString();
    static const auto logsDir = APath("logs");
    logsDir.makeDirs();
    const auto now = std::chrono::system_clock::now();
    AFileOutputStream(logsDir / "{}.0query.json"_format(now)) << query.toStdString();

    ALOG_TRACE(LOG_TAG) << "QueryStreaming: " << query;
    auto result = _new<IOpenAIChat::StreamingResponse>();

    // Subscribe to response changes and print incrementally to the TUI.
    auto printer = std::make_shared<TuiStreamingPrinter>();
    AObject::connect(result->response.changed, AObject::GENERIC_OBSERVER,
                     [printer, weak = result.weak()] {
                         if (auto s = weak.lock()) {
                             printer->update(*s->response);
                         }
                     });

    result->completed =
        [](AString query, _<StreamingResponse> result, Params params, _<TuiStreamingPrinter> printer, std::chrono::system_clock::time_point now, AString sessionId)
        -> AFuture<> {
        const auto caller = AThread::current();
        auto processJson = [=](AJson json) {
            caller->enqueue([=, json = std::move(json)] {
                auto chunk = aui::from_json<util::openai_streaming::StreamingChunk>(json);
                auto out = result->response.writeScope();
                out->id = chunk.id;
                out->created = chunk.created;
                out->model = chunk.model;
                out->system_fingerprint = chunk.system_fingerprint;
                out->usage = chunk.usage;
                chunk.collectTo(out->choices);
            });
        };

        AByteBuffer jsonTempBuffer;
        auto parseBuffer = [&, processJson] {
            while (!jsonTempBuffer.empty()) {
                ATokenizer tokenizer(std::make_unique<AByteBufferInputStream>(jsonTempBuffer));
                AString command = tokenizer.readStringWhile([](char c) {
                    return c != '{' && c != '\n';
                });
                if (command.startsWith("data: [DONE]")) {
                    break;
                }
                AUI_DEFER {
                    const auto end = AStringView(jsonTempBuffer.data(), jsonTempBuffer.size()).find("\n\n");
                    const auto at = end == std::string::npos ? jsonTempBuffer.end() : jsonTempBuffer.begin() + end + 2;
                    jsonTempBuffer.erase(jsonTempBuffer.begin(), at);
                };

                if (!command.startsWith("data:")) {
                    continue;
                }

                auto json = AJson::fromBuffer(jsonTempBuffer.slice(command.bytes().length()));
                processJson(std::move(json));

            }
        };

        AUI_ASSERT(!sessionId.empty());
        AVector<AString> headers = {"Content-Type: application/json", "x-session-id: {}"_format(sessionId) };
        if (!params.config.endpoint.bearerKey.empty()) {
            headers << "Authorization: Bearer {}"_format(params.config.endpoint.bearerKey);
        }
        AByteBuffer rawBody;
        auto httpResponse = co_await ACurl::Builder(params.config.endpoint.baseUrl + "chat/completions")
                                               .withMethod(ACurl::Method::HTTP_POST)
                                               .withTimeout(config().requestTimeoutSecs)
                                               .withHeaders(std::move(headers))
                                               .withBody(query.toStdString())
                                               .withWriteCallback([&parseBuffer, &jsonTempBuffer, &rawBody](AByteBufferView buffer) -> size_t {
                                                   ALOG_TRACE(LOG_TAG) << "QueryStreaming piece " << buffer.toStdStringView();
                                                   jsonTempBuffer << buffer;
                                                   rawBody << buffer;
                                                   try {
                                                       parseBuffer();
                                                   } catch (const AJsonException& e) {
                                                       // "unexpected" eof, parse later
                                                   }
                                                   return buffer.size();
                                               })
                                               .runAsync();
        if (httpResponse.code != ACurl::ResponseCode::HTTP_200_OK) {
            AString body = AByteBufferView(rawBody).toStdStringView();
            ALogger::warn(LOG_TAG) << "chatStreaming: status=" << httpResponse.code << ", body=" << body;
            const auto code = (int) httpResponse.code;
            if (code == 400 || code == 422) {
                // 400/422 mean the API rejected the request's *content* (e.g. "invalid tool call arguments" or
                // "invalid JSON schema"), most often because mTemporaryContext got corrupted (malformed tool_calls
                // JSON persisted into history). Retrying the exact same request forever would just spam identical
                // failures, so surface it as an exception: AppBase's notification loop already drops the
                // (irreversibly damaged) context when it catches an AException mentioning "json" in its message.
                //
                // Deliberately excludes 401/403 (auth) and 429 (rate/quota limit, e.g. "free-models-per-day") -
                // those are about the account/quota, not the request content, and will succeed again later with
                // the *same* context once the quota resets or credentials are fixed. Dropping context for those
                // would destroy conversation memory for no reason.
                throw AException("chatStreaming: HTTP {} (bad request, dropping json context) - {}"_format(code, body));
            }
        }
        // finalize
        parseBuffer();

        // ensure we delivered all events before finishing the coroutine.
#if AUI_DEBUG
        AUI_ASSERT(AThread::current() == caller);
#endif
        AThread::processMessages();
        printer->finish();
        if (auto promptTokensDetails = result->response->prompt_tokens_details.asObjectOpt()) {
            if (auto cacheWriteTokens = (*promptTokensDetails)["cache_write_tokens"].asLongIntOpt()) {
                if (*cacheWriteTokens == 0) {
                    ALogger::warn(LOG_TAG) << "Response states that no tokens were written to cache! Provider: \"" << result->response->provider << "\"";
                }
            }
        }
        AFileOutputStream(logsDir / "{}.1response.json"_format(now)) << AJson::toString(aui::to_json(*result->response));
    }(std::move(query), result, std::move(params), std::move(printer), now, messages.sessionId);
    return result;
}

AFuture<std::valarray<double>> OpenAIChatImpl::embedding(Params params, AString input) {
    ALOG_TRACE(LOG_TAG) << "embedding";
    if (input.empty()) {
        input = " ";
    }
    AVector<AString> headers = {"Content-Type: application/json"};
    if (!params.config.endpoint.bearerKey.empty()) {
        headers << "Authorization: Bearer {}"_format(params.config.endpoint.bearerKey);
    }
    tryAgain:
    auto response = AJson::fromBuffer((co_await ACurl::Builder(params.config.endpoint.baseUrl + "embeddings")
                                           .withMethod(ACurl::Method::HTTP_POST)
                                           .withTimeout(config().requestTimeoutSecs)
                                           .withHeaders(headers)
                                           .withBody(AJson::toString(AJson::Object{
                                               {"model", params.config.model},
                                               {"input", input},
                                           }))
                                           .runAsync())
                                          .body);
    if (response.contains("error")) {
        auto message = AJson::toString(response["error"]);
        if (message.contains("model failed to load, this may be due to resource limitations or an internal error")) {
            // if vram is VERY low, ollama even fails to unload the previous model.
            // in ollama_setup.sh, we set OLLAMA_KEEP_ALIVE=1m, so we will just wait the ollama to unload the model,
            // then try again.
            ALogger::warn(LOG_TAG) << "Ollama model failed to load, wait and retry...";
            co_await AThread::asyncSleep(1min / 2);
            goto tryAgain;
        }
        throw AException("Ollama error: {}"_format(message));
    }

    const auto& array = response["data"][0]["embedding"].asArray();

    std::valarray result(0.0, array.size());
    for (const auto& [i, v]: array | ranges::view::enumerate) {
        result[i] = v.asNumber();
    }
    co_return result;
}

AFuture<IOpenAIChat::AudioTranscription> OpenAIChatImpl::transcribeAudio(AByteBufferView audio, AStringView format) {
    ALOG_TRACE(LOG_TAG) << "transcribeAudio: " << audio.size() << " bytes, format=" << format;
    const auto& cfg = config().llmAudioToText;

    AFormMultipart form;
    form["model"] = { .value = AString(cfg.model) };
    form["file"] = { .value = _new<AByteBufferInputStream>(audio), .filename = "audio.{}"_format(format), .mimeType = "audio/{}"_format(format) };
    form["response_format"] = { .value = AString("verbose_json") };

    AVector<AString> headers;
    if (!cfg.endpoint.bearerKey.empty()) {
        headers << "Authorization: Bearer {}"_format(cfg.endpoint.bearerKey);
    }

    auto response = co_await ACurl::Builder(cfg.endpoint.baseUrl + "audio/transcriptions")
                        .withMethod(ACurl::Method::HTTP_POST)
                        .withHeaders(std::move(headers))
                        .withMultipart(form)
                        .withTimeout(config().requestTimeoutSecs)
                        .runAsync();

    if (response.code != ACurl::ResponseCode::HTTP_200_OK) {
        throw AException("transcribeAudio: HTTP {}: {}"_format(int(response.code), AString::fromUtf8(response.body)));
    }

    co_return aui::from_json<AudioTranscription>(AJson::fromBuffer(response.body));
}
