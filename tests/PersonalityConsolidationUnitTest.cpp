//
// Unit tests for Diary::personalityConsolidation() and the new trait_signal metadata fields.
//
// These are true unit tests (no live LLM/network): IOpenAIChat is mocked via OpenAIMock, and every
// trait_signal entry used here has a pre-populated embedding, so the code under test never needs to call
// embedding() at all. chatStreaming() is only ever expected to fire when a cluster actually qualifies
// (corroborated over time/confidence, or a single flashbulb-intensity signal) - this is the "evidence
// gating" the whole feature hinges on, so it's the main thing worth locking down here.
//

#include "AUI/IO/AFileInputStream.h"
#include "Diary.h"
#include "OpenAIMock.h"
#include "config.h"
#include "prompts.h"
#include "util/await_synchronously.h"

#include <gmock/gmock.h>

using namespace testing;

namespace {

AArc<IOpenAIChat::StreamingResponse> makeFinalResponse(const AString& content) {
    IOpenAIChat::Message msg;
    msg.role = IOpenAIChat::Message::Role::ASSISTANT;
    msg.content = content;

    IOpenAIChat::Response resp;
    resp.choices = {
        IOpenAIChat::Response::Choice{
            .index = 0,
            .message = std::move(msg),
            .finish_reason = "stop",
        },
    };
    auto result = _new<IOpenAIChat::StreamingResponse>();
    result->response.raw = std::move(resp);
    result->completed.supplyValue();
    return result;
}

Diary::EntryEx makeTraitSignal(AString id, AString trait, float valence, float intensity,
                                std::valarray<double> embedding, AString subject = "", float confidence = 0.f) {
    return Diary::EntryEx{
        .id = std::move(id),
        .metadata = {
            .confidence = confidence,
            .embedding = std::move(embedding),
            .kind = "trait_signal",
            .trait = std::move(trait),
            .valence = valence,
            .intensity = intensity,
            .subject = std::move(subject),
        },
        .freeformBody = "self-observation",
    };
}

AString readFile(const APath& path) {
    return AString::fromUtf8(AByteBuffer::fromStream(AFileInputStream(path)));
}

} // namespace

// =====================================================================
// Diary::EntryEx::Metadata JSON round-trip for the new trait_signal fields
// =====================================================================

TEST(PersonalityConsolidationUnit, MetadataRoundTripNewFields) {
    auto entries = AVector<Diary::Entry>{
        { .id = "sig1",
          .text = R"(---
{"score": 0.0, "confidence": 0.2, "lastUsed": "never", "usageCount": 0, "embedding": [], "kind": "trait_signal", "trait": "trust", "valence": -0.8, "intensity": 0.95, "subject": "12345"}
---
I noticed I got badly burned by someone I trusted.)" }
    };
    auto result = Diary::parse(entries);

    ASSERT_EQ(result.size(), 1);
    const auto& e = result.front();
    EXPECT_EQ(e.metadata.kind, "trait_signal");
    EXPECT_EQ(e.metadata.trait, "trust");
    EXPECT_FLOAT_EQ(e.metadata.valence, -0.8f);
    EXPECT_FLOAT_EQ(e.metadata.intensity, 0.95f);
    EXPECT_EQ(e.metadata.subject, "12345");
}

TEST(PersonalityConsolidationUnit, MetadataWithoutNewFieldsStillParses) {
    // old diary entries written before this feature existed have none of these fields - they must keep parsing.
    auto entries = AVector<Diary::Entry>{
        { .id = "old1", .text = "---\n{\"score\": 0.0, \"lastUsed\": \"never\", \"usageCount\": 0, \"embedding\": []}\n---\nAn ordinary memory." }
    };
    auto result = Diary::parse(entries);

    ASSERT_EQ(result.size(), 1);
    EXPECT_TRUE(result.front().metadata.kind.empty());
    EXPECT_TRUE(result.front().metadata.trait.empty());
    EXPECT_TRUE(result.front().metadata.subject.empty());
    EXPECT_FLOAT_EQ(result.front().metadata.valence, 0.f);
    EXPECT_FLOAT_EQ(result.front().metadata.intensity, 0.f);
}

// =====================================================================
// Diary::personalityConsolidation() - evidence gating
// =====================================================================

TEST(PersonalityConsolidationUnit, EmptyDiaryDoesNothing) {
    APath("test_data_personality_empty").removeFileRecursive();
    auto openAI = _new<OpenAIMock>();
    Diary diary({ .diaryDir = "test_data_personality_empty", .openAI = openAI });

    EXPECT_CALL(*openAI, embedding(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*openAI, chatStreaming(testing::_, testing::_)).Times(0);

    util::await_synchronously(diary.personalityConsolidation());
}

TEST(PersonalityConsolidationUnit, UncorroboratedWeakSignalDoesNotConsolidate) {
    APath("test_data_personality_weak").removeFileRecursive();
    auto openAI = _new<OpenAIMock>();
    Diary diary({ .diaryDir = "test_data_personality_weak", .openAI = openAI });

    // a single low-intensity, low-confidence observation - neither corroborated (needs >= 3 similar,
    // spanning >= 7 days) nor a flashbulb (needs intensity >= ~0.9). Must not touch the LLM at all.
    auto entry = makeTraitSignal("1000000000", "patience", -0.2f, 0.1f, { 1.0, 0.0 });
    diary.save(entry);

    EXPECT_CALL(*openAI, embedding(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*openAI, chatStreaming(testing::_, testing::_)).Times(0);

    util::await_synchronously(diary.personalityConsolidation());
}

TEST(PersonalityConsolidationUnit, FlashbulbSignalConsolidatesImmediately) {
    APath("test_data_personality_flash").removeFileRecursive();
    auto openAI = _new<OpenAIMock>();
    Diary diary({ .diaryDir = "test_data_personality_flash", .openAI = openAI });

    // a single, very intense observation should not need to wait for repetition ("she got burned once").
    auto entry = makeTraitSignal("1000000001", "trust", -0.9f, 0.97f, { 1.0, 0.0 });
    diary.save(entry);

    const AString kNewGrowth = "She's learned to be more careful about trusting new people right away.";
    EXPECT_CALL(*openAI, embedding(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*openAI, chatStreaming(testing::_, testing::_)).WillOnce(Return(makeFinalResponse(kNewGrowth)));

    util::await_synchronously(diary.personalityConsolidation());

    EXPECT_EQ(readFile("prompts/character_growth.md"), kNewGrowth);
    // the observation should be internalized (consumed), not left lying around to re-trigger next time.
    EXPECT_FALSE(APath("test_data_personality_flash/1000000001.md").isRegularFileExists());
}

TEST(PersonalityConsolidationUnit, CorroboratedClusterConsolidates) {
    APath("test_data_personality_corroborated").removeFileRecursive();
    auto openAI = _new<OpenAIMock>();
    Diary diary({ .diaryDir = "test_data_personality_corroborated", .openAI = openAI });

    // three similar, moderate-confidence observations about the same subject, spread over 10 days -
    // enough real-world repetition to count, even though none is a flashbulb on its own.
    const std::valarray<double> sameEmbedding{ 0.0, 1.0 };
    const auto day = 60 * 60 * 24;
    diary.save(makeTraitSignal("2000000000", "warmth", 0.5f, 0.2f, sameEmbedding, "555", 0.6f));
    diary.save(makeTraitSignal("{}"_format(2000000000 + 5 * day), "warmth", 0.5f, 0.2f, sameEmbedding, "555", 0.6f));
    diary.save(makeTraitSignal("{}"_format(2000000000 + 10 * day), "warmth", 0.5f, 0.2f, sameEmbedding, "555", 0.6f));

    const AString kNewGrowth = "She's grown noticeably warmer toward this person over time.";
    EXPECT_CALL(*openAI, embedding(testing::_, testing::_)).Times(0);
    EXPECT_CALL(*openAI, chatStreaming(testing::_, testing::_)).WillOnce(Return(makeFinalResponse(kNewGrowth)));

    util::await_synchronously(diary.personalityConsolidation());

    EXPECT_EQ(readFile("prompts/character_growth.md"), kNewGrowth);
}

TEST(PersonalityConsolidationUnit, InvalidEmptyOutputSkipsWriteAndKeepsOldGrowth) {
    APath("test_data_personality_invalid_empty").removeFileRecursive();
    auto openAI = _new<OpenAIMock>();
    Diary diary({ .diaryDir = "test_data_personality_invalid_empty", .openAI = openAI });

    saveCharacterGrowth("OLD GROWTH - MUST SURVIVE");

    auto entry = makeTraitSignal("1000000002", "trust", -0.9f, 0.99f, { 1.0, 0.0 });
    diary.save(entry);

    // model returns garbage/empty output - this cycle must be skipped, not "rolled back" (there is no
    // rollback by design), just left untouched.
    EXPECT_CALL(*openAI, chatStreaming(testing::_, testing::_)).WillOnce(Return(makeFinalResponse("   ")));

    util::await_synchronously(diary.personalityConsolidation());

    EXPECT_EQ(readFile("prompts/character_growth.md"), "OLD GROWTH - MUST SURVIVE");
}

TEST(PersonalityConsolidationUnit, OversizedOutputSkipsWriteAndKeepsOldGrowth) {
    APath("test_data_personality_invalid_oversized").removeFileRecursive();
    auto openAI = _new<OpenAIMock>();
    Diary diary({ .diaryDir = "test_data_personality_invalid_oversized", .openAI = openAI });

    saveCharacterGrowth("OLD GROWTH - MUST SURVIVE 2");

    auto entry = makeTraitSignal("1000000003", "trust", -0.9f, 0.99f, { 1.0, 0.0 });
    diary.save(entry);

    AString tooLong;
    for (size_t i = 0; i < config().personalityGrowthMaxChars + 100; i++) {
        tooLong += "x";
    }
    EXPECT_CALL(*openAI, chatStreaming(testing::_, testing::_)).WillOnce(Return(makeFinalResponse(tooLong)));

    util::await_synchronously(diary.personalityConsolidation());

    EXPECT_EQ(readFile("prompts/character_growth.md"), "OLD GROWTH - MUST SURVIVE 2");
}
