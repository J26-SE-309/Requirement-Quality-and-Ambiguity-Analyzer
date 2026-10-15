"""Context-aware reference resolution using SBERT and linguistic evidence."""

from typing import ClassVar

from quality_ml.ambiguity_checks import (

    NLP,

    find_reference_expressions,

    get_ambiguity_signals,

)

from quality_ml.competition import candidate_competition

from quality_ml.lexical_ambiguity import analyze_word_senses

from quality_ml.interpretations import generate_interpretations

from quality_ml.ranking import (

    calculate_combined_score,

    calculate_linguistic_ranking_score,

    calculate_proximity_score,

    calculate_recency_score,

    calculate_syntactic_score,

)

from quality_ml.semantic_similarity import (

    rank_candidates,

    similarity_gap,

)


class Task2AmbiguityAnalyzer:

    """Detect and rank ambiguous references.

    The analyzer supports:

    - pronouns

    - demonstratives

    - relative references

    Candidate ranking combines:

    - SBERT semantic similarity

    - token proximity

    - syntactic role

    - number/animacy compatibility

    - recency

    """

    AMBIGUITY_COMPETITION_THRESHOLD = 0.95

    PERSONAL_PRONOUNS: ClassVar[set[str]] = {

        "he",

        "him",

        "his",

        "she",

        "her",

        "hers",

        "they",

        "them",

        "their",

        "theirs",

        "it",

        "its",

        "itself",

        "themselves",

    }

    def _find_candidate_metadata(

        self,

        ranked_candidate: dict,

        candidates: list[dict],

    ) -> dict | None:
        """Find linguistic metadata for a ranked SBERT candidate."""

        for candidate in candidates:

            if (

                candidate["text"] == ranked_candidate["candidate"]

                and candidate["start"]

                == ranked_candidate["candidate_start"]

            ):

                return candidate

        return None

    def _get_candidate_position(

        self,

        candidate: dict | None,

        reference_position: int,

    ) -> int:
        """Return the candidate root token position."""

        if candidate is None:

            return reference_position

        return candidate.get(

            "root_position",

            reference_position,

        )

    def _get_candidate_dependency(

        self,

        candidate: dict | None,

    ) -> str:
        """Return the candidate's dependency relation."""

        if candidate is None:

            return ""

        return candidate.get(

            "dep",

            candidate.get("dependency", ""),

        )

    def _get_compatibility_score(

        self,

        candidate: dict | None,

    ) -> float:
        """Read compatibility computed by ambiguity_checks.

        Unknown compatibility remains neutral rather than being

        treated as incompatible.

        """

        if candidate is None:

            return 0.5

        compatibility = candidate.get(

            "compatibility",

            0.5,

        )

        if isinstance(compatibility, dict):

            return float(

                compatibility.get(

                    "overall",

                    0.5,

                )

            )

        try:

            return float(compatibility)

        except (TypeError, ValueError):

            return 0.5

    def _relative_attachment_bonus(

        self,

        reference: dict,

        candidate: dict | None,

    ) -> float:

        # Relative references such as "who" and "which" usually attach

        # to a nearby antecedent through a relative clause.

        if reference.get("kind") != "relative":

            return 0.0

        if candidate is None:

            return 0.0

        reference_head_position = reference.get("head_position")

        candidate_root_position = candidate.get("root_position")

        if (

            reference_head_position is None

            or candidate_root_position is None

        ):

            return 0.0

        dependency_proximity = candidate.get(

            "dependency_proximity",

            0.0,

        )

        if dependency_proximity >= 0.5:

            return 1.0

        if dependency_proximity >= (1.0 / 3.0):

            return 0.9

        distance = abs(

            reference_head_position

            - candidate_root_position

        )

        if distance <= 2:

            return 0.7

        if distance <= 4:

            return 0.4

        return 0.0

    def _apply_reference_rules(

        self,

        reference: dict,

        candidate: dict | None,

        compatibility_score: float,

    ) -> float:
        """Apply conservative reference-specific compatibility rules.

        These rules prevent obviously impossible candidates from

        defeating linguistically valid candidates merely because SBERT

        gives them a high semantic similarity.

        """

        if candidate is None:

            return compatibility_score

        reference_text = reference["text"].lower()

        is_human = candidate.get("is_human")

        is_plural = candidate.get("is_plural")

        # Singular human pronouns require human antecedents.

        if reference_text in {

            "he",

            "him",

            "his",

            "she",

            "her",

            "hers",

        } and is_human is False:

            return 0.0

        # Singular inanimate pronouns normally reject clearly human

        # candidates.

        if reference_text in {

            "it",

            "its",

            "itself",

        } and is_human is True:

            return 0.0

        # Plural personal pronouns should prefer plural candidates.

        if reference_text in {

            "they",

            "them",

            "their",

            "theirs",

            "themselves",

        }:

            if is_plural is False:

                return 0.0

            if is_plural is True:

                compatibility_score = max(

                    compatibility_score,

                    1.0,

                )

        # Human relative pronouns require a human candidate.

        if reference_text in {

            "who",

            "whom",

            "whose",

        } and is_human is False:

            return 0.0

        return compatibility_score

    def _rank_reference_candidates(

        self,

        text: str,

        reference: dict,

        signals: dict,

    ) -> list[dict]:
        """Generate and rank antecedent interpretations."""

        interpretations = generate_interpretations(

            text,

            reference,

            signals["candidates"],

        )

        ranked_candidates = rank_candidates(

            text,

            interpretations,

        )

        for candidate in ranked_candidates:

            metadata = self._find_candidate_metadata(

                candidate,

                signals["candidates"],

            )

            candidate_position = self._get_candidate_position(

                metadata,

                reference["position"],

            )

            proximity_score = calculate_proximity_score(

                reference["position"],

                candidate_position,

            )

            recency_score = calculate_recency_score(

                reference["position"],

                candidate_position,

            )

            dependency = self._get_candidate_dependency(

                metadata

            )

            syntactic_score = calculate_syntactic_score(

                dependency

            )

            compatibility_score = (

                self._get_compatibility_score(

                    metadata

                )

            )

            compatibility_score = (

                self._apply_reference_rules(

                    reference,

                    metadata,

                    compatibility_score,

                )

            )

            relative_bonus = (

                self._relative_attachment_bonus(

                    reference,

                    metadata,

                )

            )

            # Relative pronouns such as who/which strongly benefit

            # from local syntactic attachment information.

            if reference.get("kind") == "relative":

                syntactic_score = relative_bonus

            base_score = calculate_combined_score(

                semantic_score=candidate["similarity"],

                proximity_score=proximity_score,

                syntactic_score=syntactic_score,

            )

            # Personal pronouns need stronger recency evidence after

            # incompatible candidates have been removed.

            # Example:

            #   "The project manager ... with the software engineer

            #    who updated the document before she submitted it."

            # Both manager and engineer are human, but engineer is the

            # more recent compatible discourse candidate for "she".

            reference_text = reference["text"].lower()

            if reference_text in self.PERSONAL_PRONOUNS:

                recency_weight = 0.15

            else:

                recency_weight = 0.05

            ranking_score = (

                calculate_linguistic_ranking_score(

                    base_score=base_score,

                    compatibility_score=compatibility_score,

                    recency_score=recency_score,

                    recency_weight=recency_weight,

                )

            )

            candidate["proximity"] = proximity_score

            candidate["syntactic_score"] = syntactic_score

            candidate["compatibility_score"] = (

                compatibility_score

            )

            candidate["recency_score"] = recency_score

            # Keep combined_score compatible with the public API.

            # It now represents the final ranking score.

            candidate["combined_score"] = ranking_score

            # Internal/debugging score used by Task 2 tests.

            candidate["ranking_score"] = ranking_score

        ranked_candidates.sort(

            key=lambda candidate: candidate[

                "ranking_score"

            ],

            reverse=True,

        )

        for rank, candidate in enumerate(

            ranked_candidates,

            start=1,

        ):

            candidate["rank"] = rank

        return ranked_candidates

    def _calculate_competition_score(

        self,

        ranked_candidates: list[dict],

    ) -> float:
        """Measure competition between the top valid candidates."""

        valid_candidates = [

            candidate

            for candidate in ranked_candidates

            if candidate.get(

                "compatibility_score",

                0.5,

            ) > 0.10

        ]

        if len(valid_candidates) < 2:

            return 0.0

        first_score = valid_candidates[0][

            "ranking_score"

        ]

        second_score = valid_candidates[1][

            "ranking_score"

        ]

        difference = max(

            0.0,

            first_score - second_score,

        )

        return max(

            0.0,

            min(

                1.0,

                1.0 - difference,

            ),

        )

    def analyze(self, text: str) -> dict:
        """Analyze references and return ranked interpretations."""

        if not text or not text.strip():

            raise ValueError(

                "Requirement text must not be empty."

            )

        references = find_reference_expressions(

            text

        )

        analyses = []

        for reference in references:

            signals = get_ambiguity_signals(

                text,

                reference["position"],

            )

            ranked_candidates = (

                self._rank_reference_candidates(

                    text,

                    reference,

                    signals,

                )

            )

            exp04_evidence = candidate_competition(

                ranked_candidates, score_key="similarity"

            )

            candidate_count = len(

                ranked_candidates

            )

            # Only linguistically plausible candidates participate

            # in the final ambiguity decision.

            valid_candidates = [

                candidate

                for candidate in ranked_candidates

                if candidate.get(

                    "compatibility_score",

                    0.5,

                ) > 0.10

            ]

            if valid_candidates:

                best_candidate = valid_candidates[0]

                confidence = best_candidate[

                    "ranking_score"

                ]

            else:

                best_candidate = None

                confidence = 0.0

            competition_score = (

                self._calculate_competition_score(

                    ranked_candidates

                )

            )

            # Keep raw SBERT similarity gap as an explanatory metric.

            # The ambiguity decision uses the validation-selected EXP-04

            # margin; the legacy competition score is explanatory only.

            gap = similarity_gap(

                ranked_candidates

            )

            valid_candidate_count = len(

                valid_candidates

            )

            is_ambiguous = (

                exp04_evidence["candidate_count"] >= 2

                and exp04_evidence["top2_margin"] is not None

                and exp04_evidence["top2_margin"] <= 0.020

            )

            # Public linguistic information.

            # These names intentionally match the existing tests/API

            # expectations.

            linguistic = {

                "part_of_speech": reference.get(

                    "part_of_speech"

                ),

                "dependency": reference.get(

                    "dependency"

                ),

                "head": reference.get(

                    "head"

                ),

                "head_position": reference.get(

                    "head_position"

                ),

            }

            analyses.append(

                {

                    "reference": reference["text"],

                    "reference_kind": reference[

                        "kind"

                    ],

                    "position": reference[

                        "position"

                    ],

                    "span": {

                        "start": reference["start"],

                        "end": reference["end"],

                    },

                    "linguistic": linguistic,

                    "candidate_count": (

                        candidate_count

                    ),

                    "is_ambiguous": (

                        is_ambiguous

                    ),

                    "signals": {

                        "lexical": signals[

                            "signals"

                        ]["lexical"],

                        "syntactic": signals[

                            "signals"

                        ]["syntactic"],

                        "semantic": (

                            valid_candidate_count

                            >= 2

                        ),

                        "anaphoric": signals[

                            "signals"

                        ]["anaphoric"],

                    },

                    "candidate_interpretations": (

                        ranked_candidates

                    ),

                    "similarity_gap": gap,

                    "top2_margin": exp04_evidence["top2_margin"],

                    "normalized_entropy": exp04_evidence["normalized_entropy"],

                    "best_candidate": (

                        best_candidate["candidate"]

                        if best_candidate

                        else None

                    ),

                    "confidence": confidence,

                    "competition_score": (

                        competition_score

                    ),

                }

            )

        ambiguous_count = sum(

            analysis["is_ambiguous"]

            for analysis in analyses

        )

        ambiguity_score = (

            ambiguous_count / len(analyses)

            if analyses

            else 0.0

        )

        # Backward compatibility for older callers/tests that still

        # expect a "pronouns" field.

        pronoun_analyses = [

            analysis

            for analysis in analyses

            if analysis["reference_kind"]

            == "pronoun"

        ]

        lexical_analyses = analyze_word_senses(text, NLP)

        return {

            "is_ambiguous": (

                ambiguous_count > 0

            ),

            "score": ambiguity_score,

            "references": analyses,

            "pronouns": pronoun_analyses,
            "lexical_ambiguities": lexical_analyses,

        }
