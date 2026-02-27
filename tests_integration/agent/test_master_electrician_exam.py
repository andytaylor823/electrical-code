"""2026 Master Electrician Practice Exam -- agent integration tests.

Each test poses a question from the exam to the NEC agent, then uses an
LLM-as-judge to evaluate whether the agent's answer matches the known
correct answer.  Questions are asked open-ended (no multiple choice) so
the agent must derive the answer from RAG retrieval alone.

Run:
    pytest tests_integration/ -v
    pytest tests_integration/ -v -k "q03"       # run a single question
    pytest tests_integration/ -v --tb=short      # compact tracebacks

Correct answers are noted in comments beside each case but are NOT
provided to the agent.

See here for the full questions and answers:
https://www.tests.com/practice/Master-Electrician-Practice-Test
"""

import logging

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exam questions: (id, question_text, correct_answer_for_judge)
#
# The correct answer string is passed only to the LLM judge, never to the
# agent.  Multiple-choice options are preserved in comments for reference.
# ---------------------------------------------------------------------------

EXAM_CASES = [
    # ------------------------------------------------------------------
    # Q1  Section 220.82(A)
    #             a. multifamily dwelling units only
    #             b. dwelling units with over 3,000 sq. ft. of usable space
    # correct --> c. dwelling units supplied by service conductors with an ampacity of 100 amperes or greater
    #             d. dwelling units with a 208Y/120-volt, 3-phase, electrical system only
    # ------------------------------------------------------------------
    (
        "q01",
        "According to the NEC, the optional method of calculation for dwelling units is reserved for what type of dwelling units?",
        "Dwelling units supplied by service-entrance or feeder conductors with an ampacity of 100 amperes or greater (Section 220.82(A)).",
    ),
    # ------------------------------------------------------------------
    # Q2  Table 220.103
    #             a. 60 percent
    # correct --> b. 75 percent
    #             c. 80 percent
    #             d. 90 percent
    # ------------------------------------------------------------------
    (
        "q02",
        (
            "When calculating the service-entrance conductors of a farm where all the buildings, including the dwelling, "
            "are supplied from a common service, what percentage shall the second largest individual load in the group be calculated at?"
        ),
        "75 percent, per Table 220.103.",  # 75 percent
    ),
    # ------------------------------------------------------------------
    # Q3  Table 511.3(C)
    # correct --> a. Class I, Division 1
    #             b. Class I, Division 2
    #             c. Class II, Division 1
    #             d. Class II, Division 2
    # ------------------------------------------------------------------
    (
        "q03",
        (
            "What is the hazardous location classification of the entire area of a service pit located in a "
            "commercial major repair garage that is not provided with mechanical ventilation?"
        ),
        "Class I, Division 1, per Table 511.3(C).",
    ),
    # ------------------------------------------------------------------
    # Q4  Section 501.15(A)(1)
    #             a. 6 inches
    #             b. 24 inches
    #             c. 12 inches
    # correct --> d. 18 inches
    # ------------------------------------------------------------------
    (
        "q04",
        (
            "What is the MAXIMUM distance conduit seals are permitted to be installed from an enclosure that "
            "houses devices that produce arcs or high temperatures in a Class I, Division 1 location?"
        ),
        "18 inches, per Section 501.15(A)(1).",
    ),
    # ------------------------------------------------------------------
    # Q5  Section 404.8(B)
    #             a. 150 volts
    #             b. 250 volts
    # correct --> c. 300 volts
    #             d. 240 volts
    # ------------------------------------------------------------------
    (
        "q05",
        "Where three switches are grouped or ganged in an outlet box without permanent barriers, the voltage between the adjacent switches shall NOT exceed what value?",
        "300 volts, per Section 404.8(B).",
    ),
    # ------------------------------------------------------------------
    # Q6  Section 314.28 / Table 314.16(B)
    #             a. 5.00 cu. in.
    #             b. 5.50 cu. in.
    #             c. 6.25 cu. in.
    # correct --> d. None of these
    # ------------------------------------------------------------------
    (
        "q06",
        (
            "In order to determine the adequate size of a junction box containing size 4 AWG THWN copper conductors, "
            "what is the volume allowance in cubic inches required per conductor?"
        ),
        (
            "None of the standard volume allowances apply. For conductors 4 AWG and larger, "
            "box sizing is determined by Section 314.28 based on raceway diameter, not Table 314.16(B)."
        ),
    ),
    # ------------------------------------------------------------------
    # Q7  Table 300.5, Column 3
    #             a. 12 inches
    # correct --> b. 18 inches
    #             c. 24 inches
    #             d. 30 inches
    # ------------------------------------------------------------------
    (
        "q07",
        (
            "You are to install an underground run of trade size 2 in. Schedule 40 PVC conduit containing "
            "120/240-volt, single-phase service-entrance conductors supplying a one-family dwelling. The PVC "
            "will not cross under any public streets, roads, driveways, or alleys. What is the minimum burial "
            "depth from final grade?"
        ),
        "18 inches, per Table 300.5, Column 3.",
    ),
    # ------------------------------------------------------------------
    # Q8  Section 230.26
    # correct --> a. 10 feet
    #             b. 12 feet
    #             c. 15 feet
    #             d. 18 feet
    # ------------------------------------------------------------------
    (
        "q08",
        (
            "Where an eye-bolt is provided as a means of attachment of 120/240-volt, single-phase residential "
            "service-drop conductors, what is the minimum height above finished grade for this point of attachment?"
        ),
        "10 feet, per Section 230.26.",
    ),
    # ------------------------------------------------------------------
    # Q9  Table C.9 (Annex C)
    #             a. 2½ in.
    # correct --> b. 3 in.
    #             c. 3½ in.
    #             d. 4 in.
    # ------------------------------------------------------------------
    (
        "q09",
        (
            "You are to install four (4) size 350 kcmil THWN copper conductors in a run of rigid metal conduit (RMC). "
            "What is the MINIMUM trade size RMC required to enclose the conductors?"
        ),
        "3 inches (trade size 3 in. RMC), per Table C.9 of Annex C.",
    ),
    # ------------------------------------------------------------------
    # Q10  Table 250.122
    #             a. 2 AWG
    #             b. 4 AWG
    #             c. 6 AWG
    # correct --> d. 8 AWG
    # ------------------------------------------------------------------
    (
        "q10",
        (
            "Where size 2 AWG THWN/THHN copper conductors, protected by a 100-ampere rated circuit breaker, are "
            "enclosed in a Schedule 40 PVC conduit to supply an air-conditioning unit, what is the MINIMUM size "
            "copper equipment grounding conductor required?"
        ),
        "8 AWG copper, per Table 250.122 based on 100-ampere overcurrent protection.",
    ),
    # ------------------------------------------------------------------
    # Q11  Section 240.24(A) / Section 404.8(A)
    #             a. 6 ft.
    #             b. 6 ft., 6 in.
    # correct --> c. 6 ft., 7 in.
    #             d. 7 ft.
    # ------------------------------------------------------------------
    (
        "q11",
        (
            "As a general rule, switches not over 1,000 volts containing fuses and circuit breakers shall be "
            "readily accessible and installed so that the center of the grip of the operating handle of the "
            "switch or circuit breaker, when in its highest position, is not more than how far above the floor "
            "or working platform?"
        ),
        "6 feet, 7 inches (6 ft. 7 in.), per Section 240.24(A) and Section 404.8(A).",
    ),
    # ------------------------------------------------------------------
    # Q12  Section 300.5(D)(3)
    #             a. 6 inches
    #             b. 10 inches
    #             c. 8 inches
    # correct --> d. 12 inches
    # ------------------------------------------------------------------
    (
        "q12",
        (
            "Underground service conductors of not over 1,000 volts that are not encased in concrete and that "
            "are buried 18 inches or more below grade shall have their location identified by a warning ribbon "
            "placed at least how far above the underground installation?"
        ),
        "12 inches, per Section 300.5(D)(3).",
    ),
    # ------------------------------------------------------------------
    # Q13  Section 314.23(F)
    #             a. 12 inches
    # correct --> b. 18 inches
    #             c. 15 inches
    #             d. 24 inches
    # ------------------------------------------------------------------
    (
        "q13",
        (
            "Where the sole support of an outdoor-installed Type FS box that contains devices or supports luminaires "
            "is two underground buried threaded intermediate metal conduits (IMC) or rigid metal conduits (RMC) that "
            "emerge from the ground, how far from the point where the conduit emerges from the ground must the "
            "conduits be secured?"
        ),
        "18 inches (the conduits shall be secured within 18 inches of the box), per Section 314.23(F).",
    ),
    # ------------------------------------------------------------------
    # Q14  Section 705.31
    # correct --> a. 10 feet
    #             b. 6 feet
    #             c. 15 feet
    #             d. 20 feet
    # ------------------------------------------------------------------
    (
        "q14",
        (
            "As a general rule regarding solar photovoltaic (PV) systems, supply-side conductor connections for PV systems must terminate in an overcurrent protection device that is within no more than what "
            "distance of the service disconnecting means?"
        ),
        "10 feet, per Section 705.31 (NOTE THAT THIS SECTION IS FROM OLDER VERSIONS AND DOES NOT EXIST IN THE NEC 2023, SO CITATION MAY NOT MATCH)",
    ),
    # ------------------------------------------------------------------
    # Q15  Section 551.73 / Table 551.73(A)
    #             a. 1,120 amperes
    #             b. 639 amperes
    # correct --> c. 588 amperes
    #             d. 1,400 amperes
    # ------------------------------------------------------------------
    (
        "q15",
        (
            "Calculate the minimum demand load, in amperes, on the ungrounded service-entrance conductors for a "
            "recreational vehicle park that has 28 RV sites, all with 50-ampere outlets at each RV site. The "
            "supply system is single-phase, 120/240 volts."
        ),
        "588 amperes. Calculation: 12,000 VA x 28 sites = 336,000 VA; 336,000 VA x 0.42 demand factor (Table 551.73(A)) = 141,120 VA; 141,120 VA / 240 V = 588 A.",
    ),
    # ------------------------------------------------------------------
    # Q16  Section 501.15(A)(1)(2)
    #             a. 12 inches
    #             b. 6 inches
    #             c. 24 inches
    # correct --> d. 18 inches
    # ------------------------------------------------------------------
    (
        "q16",
        (
            "Conduit seals shall be installed at no more than what distance from the entrance of an enclosure "
            "that contains arcing or sparking equipment, or only splices, taps, or terminals, where located in "
            "Class I, Division 1 hazardous locations?"
        ),
        "18 inches, per Section 501.15(A)(1)(2).",
    ),
    # ------------------------------------------------------------------
    # Q17  Section 590.6(A)(1) / Section 590.6(B)
    #             a. I only (125-volt, 15- and 20-ampere)
    #             b. II only (250-volt, 20- and 30-ampere)
    # correct --> c. both I and II
    #             d. neither I nor II
    # ------------------------------------------------------------------
    (
        "q17",
        (
            "When building a temporary service at a construction site, which of the following receptacles must be GFCI protected: (I) 125-volt, 15- and 20-ampere receptacles, (II) 250-volt, 20- and 30-ampere receptacles, or both?"
        ),
        "Both. All 125-volt 15- and 20-ampere receptacles (Section 590.6(A)(1)) and 250-volt receptacles (Section 590.6(B)) require GFCI protection.",
    ),
    # ------------------------------------------------------------------
    # Q18  Section 250.53(A)(3)
    #             a. 4 feet
    # correct --> b. 6 feet
    #             c. 8 feet
    #             d. 10 feet
    # ------------------------------------------------------------------
    (
        "q18",
        "Where it is necessary to install more than one ground rod used as a grounding electrode, what is the minimum spacing between the ground rods?",
        "6 feet, per Section 250.53(A)(3).",
    ),
    # ------------------------------------------------------------------
    # Q19  P = I x E calculation
    #             a. 300 kVA
    #             b. 112.5 kVA
    #             c. 225 kVA
    # correct --> d. 150 kVA
    # ------------------------------------------------------------------
    (
        "q19",
        "Where a transformer supplies a single-phase, 120/240-volt, 600-ampere service, what is the minimum kVA rating the transformer should have?",
        "150 kVA. Calculation: 600 A x 240 V = 144,000 VA = 144 kVA, so the next standard size is 150 kVA. Answer of 144 kVA is also correct, as it is the literally correct value, agnostic of standardsizing.",
    ),
    # ------------------------------------------------------------------
    # Q20  Section 300.6 (MC cable / metal conduit indoor wet locations)
    #             a. 1/8 in.
    # correct --> b. 1/4 in.
    #             c. 1/2 in.
    #             d. 3/8 in.
    # ------------------------------------------------------------------
    (
        "q20",
        "Metal conduit installed in indoor wet locations must have a minimum airspace clearance of how much between the conduit and the wall or supporting surface?",
        "1/4 inch (0.25 in.), per Section 300.6 or Section 312.2",
    ),
]


# ---------------------------------------------------------------------------
# Parametrised test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "question, correct_answer",
    [(q, a) for _, q, a in EXAM_CASES],
    ids=[qid for qid, _, _ in EXAM_CASES],
)
def test_nec_exam_question(question, correct_answer, ask_agent, llm_judge, results_writer):
    """Ask the agent an NEC exam question and have an LLM judge the answer."""
    logger.info("QUESTION: %s", question)

    # Invoke the agent (real LLM + RAG call)
    agent_response = ask_agent(question)
    logger.info("AGENT RESPONSE: %s", agent_response)

    # Evaluate with LLM judge
    verdict = llm_judge(question, correct_answer, agent_response)
    logger.info("JUDGE VERDICT: passed=%s reasoning=%s", verdict.passed, verdict.reasoning)

    # Persist structured result for later inspection
    results_writer(
        {
            "question": question,
            "correct_answer": correct_answer,
            "agent_response": agent_response,
            "passed": verdict.passed,
            "reasoning": verdict.reasoning,
        }
    )

    delimiter = "=" * 100
    assert verdict.passed, (
        f"{delimiter}\n"
        f"LLM judge failed this question.\n"
        f"{delimiter}\n"
        f"  Reasoning: {verdict.reasoning}\n"
        f"{delimiter}\n"
        f"  Expected: {correct_answer}\n"
        f"{delimiter}\n"
        f"  Agent said: {agent_response[:900]}\n"
        f"{delimiter}\n"
    )
