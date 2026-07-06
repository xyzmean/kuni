//
// record_trait_signal: lets Kusi note a real self-observation, gated later by Diary::personalityConsolidation().
//

#include "record_trait_signal.h"

#include "config.h"

#include "AUI/Logging/ALogger.h"

#include <chrono>

namespace {
// Soft daily cap so a single chatty day can't flood her with self-observations. This is the only rate limit -
// deliberately simple, since the real gate against instant/arbitrary personality change is evidence-based
// corroboration in Diary::personalityConsolidation(), not this tool.
constexpr size_t MAX_TRAIT_SIGNALS_PER_DAY = 8;
}

OpenAITools::Tool tools::recordTraitSignal(_<IOpenAIChat> openAI, Diary& diary) {
    return {
        .name = "record_trait_signal",
        .description =
            "Quietly note something real you noticed about yourself just now - a moment that stung, one that made "
            "you glow, something you realized you like or can't stand, a topic from a channel you keep getting drawn "
            "back to, or a shift in how you feel about someone specific (including papik - nobody is exempt from how "
            "you actually feel). This is not for every message, only when you genuinely notice something. It does "
            "not change who you are by itself: only if the same kind of thing keeps happening over time, or if this "
            "one moment is intense enough on its own, will it slowly become part of you.",
        .parameters = {
            .properties = {
                { "observation", { .type = "string",
                                    .description = "Short first-person self-observation, e.g. \"I noticed I get "
                                                    "impatient when people repeat questions I've already "
                                                    "answered.\" or \"I keep coming back to posts about "
                                                    "astronomy.\"" } },
                { "trait", { .type = "string",
                             .description = "Short free-form tag for grouping, e.g. patience, trust, humor_style, "
                                             "boundary_setting, interest." } },
                { "valence", { .type = "number",
                               .description = "Emotional sign of the moment, from -1 (it hurt/burned you) to 1 (it "
                                               "made you genuinely happy)." } },
                { "intensity", { .type = "number",
                                 .description = "How vivid/significant this felt, from 0 to 1. A value close to 1 "
                                                 "means this alone can teach you something, without it needing to "
                                                 "happen again." } },
                { "subject", { .type = "string",
                               .description = "Chat/contact id if it's about one specific person, or a topic/channel "
                                               "name if it's about a recurring interest rather than a person. Leave "
                                               "empty if it's about people or things in general." } },
            },
            .required = { "observation", "trait", "valence", "intensity" },
        },
        .handler = [openAI = std::move(openAI), &diary](OpenAITools::Ctx ctx) -> AFuture<AString> {
            auto observation = ctx.args["observation"].asStringOpt().valueOrException("observation required");
            auto trait = ctx.args["trait"].asStringOpt().valueOrException("trait required");
            auto valence = ctx.args["valence"].asNumberOpt().valueOr(0.0);
            auto intensity = ctx.args["intensity"].asNumberOpt().valueOr(0.0);
            AString subject;
            if (auto s = ctx.args["subject"].asStringOpt()) {
                subject = *s;
            }

            const auto now =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                    .count();

            size_t signalsToday = 0;
            for (const auto& entry : diary.list()) {
                if (entry.metadata.kind != "trait_signal") {
                    continue;
                }
                if (now - entry.id.toLong().valueOr(0) <= 24 * 60 * 60) {
                    signalsToday++;
                }
            }
            if (signalsToday >= MAX_TRAIT_SIGNALS_PER_DAY) {
                co_return "You've already noted quite a few things about yourself today - let them settle before "
                          "adding more.";
            }

            Diary::EntryEx entry{
                .id = "{}"_format(now),
                .metadata = {
                    .confidence = 0.f, // a fresh self-observation is a theory about herself, not yet a fact.
                    .kind = "trait_signal",
                    .trait = trait,
                    .valence = static_cast<float>(glm::clamp(valence, -1.0, 1.0)),
                    .intensity = static_cast<float>(glm::clamp(intensity, 0.0, 1.0)),
                    .subject = subject,
                },
                .freeformBody = observation,
            };
            try {
                entry.metadata.embedding = co_await openAI->embedding({ .config = config().embedding }, observation);
            } catch (const AException& e) {
                ALogger::err("record_trait_signal") << "can't embed observation: " << e;
                co_return "Couldn't note that down right now. Try again later.";
            }
            diary.save(entry);
            co_return "Noted.";
        },
    };
}
