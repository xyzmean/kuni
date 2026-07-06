#pragma once

#include <Diary.h>
#include <IOpenAIChat.h>
#include <OpenAITools.h>
#include <functional>

namespace tools {
OpenAITools::Tool ask(std::function<AString()> additionalDetails, _<IOpenAIChat> openAI, Diary& diary);

/**
 * @brief Runs the same #ask/#query RAG loop used by the `ask` tool, without the tool-call wrapping.
 * @details
 * For headless debugging (`kuni --ask-diary "..."`) - lets you query the diary directly from a terminal,
 * without Telegram or a GUI.
 */
AFuture<AString> askDebugQuery(IOpenAIChat& openAI, Diary& diary, const AString& query);
}
