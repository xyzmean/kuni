#pragma once
#include "AUI/Common/AString.h"
#include "AUI/Common/AVector.h"
#include "AUI/Json/AJson.h"
#include "IOpenAIChat.h"
#include "MetricsBreadcumbs.h"

struct OpenAITools {
    struct Ctx {
        OpenAITools& tools;
        AJson args;
        const AVector<IOpenAIChat::Message::ToolCall>& allToolCalls;
    };
    using Handler = std::function<AFuture<AString>(Ctx ctx)>;

    struct Tool {
        AString type = "function";
        AString name;
        AString description;
        struct Parameters {
            AString type = "object";
            struct Property {
                AString type = "string";
                AString description;
            };
            AMap<AString, Property> properties;
            AVector<AString> required; // required properties
            bool additionalProperties = false;
        } parameters;
        // NOTE: kept false because most tools (e.g. send_telegram_message) intentionally declare optional
        // properties that aren't listed in `required` (photo_filename, reply_to_message_id, etc.). OpenAI's
        // strict mode demands every property be required (using nullable types for the truly-optional ones),
        // and providers that actually enforce this (Groq does; ollama.com/openrouter didn't) reject the tool's
        // JSON schema outright with HTTP 400 "invalid JSON schema for tool ...", breaking every single request.
        bool strict = false;
        Handler handler;
    };

    OpenAITools(std::initializer_list<Tool> tools);

    /**
     * @brief Optional hook fired after each tool call handler completes successfully.
     * Not called if the handler throws. Set by AppBase::updateTools to emit AppBase::toolCallFired.
     */
    std::function<void(const AString& toolName)> onAfterToolCall;

    AFuture<IOpenAIChat::Session> handleToolCalls(const AVector<IOpenAIChat::Message::ToolCall>& toolCalls, const _<MetricsBreadcumbs>& metricsBreadCumbs = nullptr);

    AJson asJson() const;

    [[nodiscard]] AMap<AString, Tool> handlers() const { return mHandlers; }

    void insert(Tool tool) { mHandlers[tool.name] = std::move(tool); }

private:
    AMap<AString, Tool> mHandlers;
};
