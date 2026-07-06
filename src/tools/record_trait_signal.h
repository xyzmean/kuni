#pragma once
#include <Diary.h>
#include <IOpenAIChat.h>
#include <OpenAITools.h>

namespace tools {
/**
 * @brief Lets her jot down a real self-observation - not a personality edit, just a moment worth remembering.
 * @details
 * Writes a "trait_signal" diary entry (see Diary::EntryEx::Metadata::kind). On its own this changes nothing -
 * only Diary::personalityConsolidation() can turn a pattern of these (or one intense enough one) into an
 * actual, slow shift of who she is.
 */
OpenAITools::Tool recordTraitSignal(_<IOpenAIChat> openAI, Diary& diary);
}
