#pragma once

#include "AUI/IO/APath.h"

#include <AUI/Common/AString.h>

struct Prompts {
    static AString load(const APath& path, AStringView defaultPrompt);
    AString system;
    AString characterBase;
    AString characterAppearance;
    AString photoToText;
    AString stickerToText;
    AString antiRepeatPrompt;
    AString diarySave;
    AString sleepConsolidator;
    AString recordAudioSpeech;
    AString characterGrowth;
    AString personalityConsolidator;
};

const Prompts& prompts();

/**
 * @brief Overwrites prompts/character_growth.md with new content.
 * @details
 * There is deliberately no backup/versioning here - see Diary::personalityConsolidation(). The file is
 * hot-reloaded via the same file watcher as every other prompt file, so `prompts().characterGrowth` picks
 * up the change automatically.
 */
void saveCharacterGrowth(AStringView newContent);
