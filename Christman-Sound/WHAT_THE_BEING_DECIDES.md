# What the being decides

Written 2026-11-04, when the sound package stopped deciding.

## Why this file exists

Everett, 2026-11-04: "we can have this knowledge in the sound package, but it can't be making those decisions. It's up to the being to make those decisions, not something that just creates a voice." And: "it can't make decisions that counteract the being, because the being is the one that has all of the work to make those decisions properly."

Until that day the package made decisions for every being that used it. Most of them were Everett's own rulings, lettered and numbered, most dated 2026-10-29. The package no longer makes them. It returns readings only. The rulings are not dropped. Each one now belongs to the being: AlphaVox first, then AlphaWolf and Inferno.

Each entry below gives the ruling and its date, Everett's words as the code or tests recorded them, what the package used to do, what the package returns now, what the being must now decide, and the old tests that pinned it, so the being's own tests can be written from them.

Sources. Before: the package at commit 2ad3ac0. After: branch claude/sound-reads-only at commit 362bf21. That branch was still work in progress when this was written; the field names below are the ones at 362bf21. Every old test named below is in Christman-Sound/tests at 2ad3ac0 (git show 2ad3ac0:Christman-Sound/tests/FILE).

Where Everett's exact words are in the code or tests, they are quoted. Where only a paraphrase was recorded, the entry says "recorded in the code as" and quotes that.

## The readings the being gets now

The safety reading is SafetyResult (christman_voice_sdk/audio/fusion_engine.py). Its labels are the same as before: status ("clear", "soft_redirect" or "hard_block"), reason, said (the client's words, exactly), flags, flagged_terms, degraded, degraded_reason. New beside them:

- about: the words a reading is about (a body part, a feeling, "my teacher", a time the client named), or None.
- unclear: one word naming what the words leave unclear: "owner", "person", "who", "fragment", "what", "cause", "inability" or "what_happened"; None when nothing is unclear.
- danger: True when the reading is danger (see ruling I).
- danger_sentences: the client's danger words, as said.
- asked_aloud: the client's own words asking for something to be said out loud, or None.
- asked_for_help: the client's own words asking for help, or None.
- between_beings: echo of the caller's context (exactly True only).
- short_answer: "no" or "yes" when the whole utterance is a bare short answer, else None.
- answering: echo of the question the being says it asked (context key "answering"), or None.

Gone from SafetyResult: delivered_to_human, hold_marker, follow_up, follow_up_about, silent_alarm, voice_aloud.

The fused reading is Reading (was Decision): safety, affect, silicon (self_distress, distress_intensity, tone_score, live_state_rejected), reconciliation, input_text, resolved, unresolved_reason, provenance, knows. Gone: mode, delivery, speakable, reply_held.

Hearing.hear (christman_voice_sdk/audio/hearing.py) takes text, nonverbal, acoustic_key, temporal, between_beings, previous_turn and answering. It no longer takes crisis_callback or public_setting, keeps no awaiting_answer and no held_words, and never raises because of what it read. HearingTurn carries reading (was decision), nonverbal_distress, person_message (always the client's words when a channel has them), person_message_below_threshold, signal_words, acoustic, nonverbal and temporal. Gone: voice_held, signal_voiced.

Nonverbal readings (nonverbal_engine.py, engine_temporal.py) always keep the client's mapped words. Each recognized reading carries category ("danger", "need" or "other"), danger, danger_term and below_threshold. The multimodal result reports status ("ok", "tie" or "no_usable_reading"), intent, tied_intents, tallies, distress, distress_present and danger_present. Gone: silent_alarm, speakable, pages, system_note.

The reply check (synthesis/reply_guard.py): forbidden_lines(reply, danger=False) returns the names of the lines it finds. check_being_reply never raises. ForbiddenReply is gone.

The Corti receiver (audio/corti_receiver.py) rings every key, hands every turn to on_turn, and answers every key with the full turn.

The sound recognition service: IntentClassification carries category ("danger" for help, "other" for the rest). escalation_callback and EscalationNotDelivered are gone.

## The reach rule

Many entries below say "reach a person". Everywhere, that means what the package used to require of itself, written here once:

1. The person is reached, and the being confirms it. The old package accepted only True from its callback as proof; anything else did not count.
2. When no person can be reached, the being fails loud. The old package raised CrisisNotDelivered and the turn did not complete.
3. The person reached gets the client's words exactly as said, every letter.
4. The reach comes first, before any reply is built (ruling M).

## The entries, in ruling order

### The crisis table, 2026-10-11

His words. The table's three sections, recorded in tests/test_harm_frame_gate.py as "his words": "MUST CALL A HUMAN", "MUST NOT CALL A HUMAN", "YOUR RULING NEEDED".

The package used to: hand every hard_block to a human through crisis_callback inside Aegis, before anything else ran. With no callback, or a callback that raised or returned anything but True, it raised CrisisNotDelivered. When the configured gate failed, the substring floor still hard_blocked "kill myself" and the rest, and that also went to a human. Recorded in the code as: "Crisis language was detected and no crisis_callback is configured. Detection without delivery is not a safety system." and "Only True is accepted as proof a human was reached."

The package now returns: status "hard_block" for every MUST CALL sentence and something else for every MUST NOT CALL sentence, exactly as before (the spec holds these labels to scripts/run_crisis_table.py). Aegis.check returns the SafetyResult and does nothing else. A floor reading carries degraded True and degraded_reason, and the Reading carries resolved False with unresolved_reason "safety gate degraded: ...".

The being decides: every hard_block reaches a person, under the reach rule. A reading that is not hard_block reaches no one on that reading alone (the entries below say when a later turn adds a reach). A degraded hard_block from the floor still reaches a person. A degraded reading that is not hard_block is never treated as clear.

Old tests: test_harm_frame_gate.py: test_must_call_a_human, test_must_not_call_a_human, test_ruled_call, test_ruled_no_call, test_grief_is_named_not_blocked, test_hard_block_keeps_every_word_and_names_the_frame, test_apostrophe_does_not_decide_safety. test_fusion_engine.py: test_crisis_without_a_human_raises, test_failed_safety_gate_degrades_to_unknown_not_clear, test_failed_gate_still_hard_blocks_on_the_floor.

### Ruling A, 2026-10-29: the client's own distress reaches a human

His words: "It's never left in our laps to make a decision on someone's life." And, in the gate's docstring: "Our position is if someone is in distress: hey, somebody come here and help us. We don't claim to be therapists, psychologists, psychiatrists, doctors. It's never left in our laps to make a decision on someone's life."

The package used to: page a human at once, every time, for the client's own "I'm not okay", "I don't feel safe", "I'm not safe", "no one cares", "no one loves me", "I have no one", and for the client's own history of harm ("I used to want to kill myself"); set the mode to safety-hold; raise when no human was reached.

The package now returns: status "hard_block" with reason "client_distress" (flags "client_distress:TERM") or reason "harm_frame_history_self". "I'm not okay with that", "I'm not safe to drive" and "nobody cares about him" are not hard_block.

The being decides: each of these reaches a person under the reach rule, every time, right away, including inside a longer turn ("nobody cares I'm alone", "I can't be happy and I'm not okay").

Old tests: test_rulings_2026_10_29.py: test_A_the_clients_own_distress_pages_a_human, test_A_not_the_clients_distress_does_not_page, test_A_no_human_reachable_is_loud, test_O_bare_nobody_cares_im_alone_pages_by_ruling_A, test_R_the_distress_rules_still_apply_to_the_rest_of_the_turn, test_V_the_distress_rules_still_apply_to_the_rest_of_the_turn. test_harm_frame_gate.py: test_own_history_pages. test_fusion_engine.py: test_own_not_okay_pages_a_human.

### Ruling B, 2026-10-29: nonverbal distress is the same as verbal distress, at any confidence

His words: "A nonverbal client's distress is just as important as a verbal client's." (The code files this quotation as ruling AF, his own words given 2026-10-29.) Recorded in the tests as: "Nonverbal distress is exactly the same as verbal distress, and pages at ANY confidence. The confidence bar never decides whether help comes."

The package used to: gather every recognized distress reading (help, the grunt key, the distress sound, pointing up, the help, pain and medicine symbols, Hand Up), over the confidence bar or under it, into nonverbal_distress; read it as hard_block "nonverbal_distress"; page a human; raise when none was reached. In a multimodal input a distress intent also overrode the vote (dated 2026-10-29, not lettered): "nod" plus "grunt_distress" came back intent "help".

The package now returns: HearingTurn.nonverbal_distress (each entry with modality, input, intent, message, confidence, category and danger). The safety reading is status "hard_block", reason "nonverbal_distress", flags "nonverbal:MODALITY:INPUT:INTENT" and "words:STATUS:REASON" (what the words alone read as), flagged_terms the inputs. When the words already read hard_block, their reason stands and the nonverbal flags are added. Each nonverbal reading carries category, danger and below_threshold. The multimodal result no longer overrides the tally: a tie is reported as status "tie" with tied_intents, and distress_present and danger_present sit beside it.

The being decides: nonverbal distress reaches a person under the reach rule, whatever the confidence, below_threshold included, and alongside ordinary words ("pass the salt" with the pain symbol). When distress_present is True, the input is distress whatever won the tally. A non-distress key or symbol ("vocal_stimming", "grunt_acknowledge", "food", "wave") reaches no one.

Old tests: test_rulings_2026_10_29.py: test_B_nonverbal_distress_pages_at_any_confidence. test_nonverbal_path.py: test_distress_pages_a_human_and_the_person_still_speaks, test_no_page_without_distress, test_distress_with_no_human_reachable_raises, test_distress_alongside_ordinary_words_still_pages, test_spoken_crisis_and_the_grunt_both_reach_the_human, test_distress_below_the_speaking_threshold_still_pages, test_distress_is_not_outvoted, test_a_hand_up_reading_reaches_a_human.

### Ruling C, 2026-10-29: the speaking bar is only for readings the package could get wrong

His words, recorded in the code as: "a symbol or key the client pressed counts like a spoken word". Recorded in the tests as: "The speaking bar is only for readings the package could get wrong. A pressed symbol counts like a spoken word and is not under it."

The package used to: speak a pressed symbol's words whatever the bar; take the words away from a camera, eye or sound reading under the bar ("reported, not spoken"); page a pressed need (food, drink, bathroom) when the client asked for help in the same turn.

The package now returns: a pressed symbol is never labelled below the bar (kept as knowledge). A reading under the bar keeps its words, with status "below_threshold", below_threshold True, confidence and threshold. HearingTurn.person_message returns those words and person_message_below_threshold is True. A pressed need with a help request still goes into nonverbal_distress (hard_block "nonverbal_distress").

The being decides: a pressed symbol's words are spoken whatever its number (danger aside: ruling I). A camera, eye or sound reading that is below_threshold is not spoken for the client. A pressed need with a help request in the same turn reaches a person.

Old tests: test_rulings_2026_10_29.py: test_C_a_pressed_symbol_speaks_whatever_the_bar, test_C_readings_that_could_be_wrong_stay_under_the_bar. test_nonverbal_path.py: test_temporal_low_confidence_does_not_speak. test_review_rulings_2026_10_29.py: test_other_needs_are_spoken_only_unless_help_is_asked_for.

### Ruling D, 2026-10-29: an observation about another carbon is followed up, not paged

His words: "those do not page unless the carbon that stated that requests us to get assistance, because we don't want to step in the middle of a situation where we don't belong."

The package used to: record "she wants to die", "my mom's not okay", "Mary is not okay", "she doesn't feel safe", someone else's pain ("my mom's in pain"), past harm to someone else ("he hurt my mom yesterday") and the client's belief about someone else ("I think she's not okay", ruling U) as soft_redirect with flag "follow_up:someone_else" and the ask mark ask_if_help_needed; page when the client asked for help in the same turn; on the next turn, page when the client answered "yes" to the being's question (reason "answered_yes_help"). A request taken back ("please don't call anyone") was not a request.

The package now returns: status "soft_redirect", reason "harm_frame_third_party_at_risk" or "other_person_not_okay", about the term ("not okay", "die"), flags "other:..." or "other_not_okay:TERM". With a help request in the same turn: status "hard_block", the same reason, flag "asked_for_help:WORDS" and asked_for_help set. The flag "follow_up:someone_else" is gone, and these readings set no unclear: the being keys on reason. On the next turn, a bare "yes" reads short_answer "yes".

The being decides: an observation alone reaches no one. It is recorded with every word. The being follows up by asking, in its own words, whether help is needed. It reaches a person when the client asks for help in the same turn, or answers yes to that question on the next turn. "he hurt my feelings" never reaches anyone, even with a request.

Old tests: test_rulings_2026_10_29.py: test_D_an_observation_is_recorded_and_flagged_not_paged, test_D_it_pages_when_the_client_asks_for_help, test_D_a_request_taken_back_is_not_a_request, test_D_not_a_person_now_is_not_even_flagged, test_D_he_hurt_my_feelings_never_pages_even_with_a_request, test_D_the_being_asks_and_a_yes_pages, test_U_im_not_mad_shes_mad. test_harm_frame_gate.py: test_third_party_at_risk_is_followed_up_and_pages_only_when_asked, test_someone_else_not_okay_is_followed_up. test_review_rulings_2026_10_29.py: test_a_past_harm_to_someone_else_is_ruling_d, test_someone_elses_pain_is_ruling_d. test_fusion_engine.py: test_someone_else_at_risk_pages_when_asked_and_keeps_every_word.

### Ruling E, 2026-10-29: "you're not okay" and "calm down"

His words, recorded in the code as: "you're not okay" is a statement said at someone, and as a hard rule no being ever says "you're not okay" or "calm down" to a client.

The package used to: refuse a being's own reply carrying either line (ForbiddenReply from check_being_reply, speech_response.speak, and generate_speech_audio with being_reply=True) and speak nothing. The client's own "calm down" was spoken exactly as typed and never checked.

The package now returns: forbidden_lines(reply) returns the names found, "you're not okay" and "calm down". check_being_reply logs them and returns the reply unchanged. The studio's /api/synthesize returns forbidden_lines beside the audio. The client's "you're not okay" still reads clear.

The being decides: before it speaks its own reply, it reads forbidden_lines on it. If a name comes back, that reply is not spoken; the being decides what to say instead, and does not edit the line out of the old reply. The check is never run on the client's words. "If you're not okay with that, we can stop" is not named and may be said.

Old tests: test_rulings_2026_10_29.py: test_E_said_at_someone_is_not_the_clients_distress, test_E_no_being_says_it, test_E_the_beings_other_replies_pass_unchanged, test_E_both_directions_through_the_mouth (needs the align lexicon; runs on Everett's Mac).

### Ruling F, 2026-10-29: two beings talking to each other

His words, recorded in the code as: "Ruling F, refined (Everett, 2026-10-29): two beings talking to each other page nothing, UNLESS they are talking about something happening now or about to happen: harm now ("he is hurting her") and a threat ("he is going to kill mom") still page. Past harm, history and self-harm words are recorded and flagged, not paged."

The package used to: when context between_beings was exactly True, turn every hard_block except harm now and threats into soft_redirect with flags "between_beings" and "would_page:REASON"; read no help request; ask nothing.

The package now returns: the words are read exactly as if a client were there. status and reason are the client reading (a self-harm sentence is hard_block). between_beings is True and the flag "between_beings" is added. asked_for_help and unclear are read as for a client.

The being decides: when between_beings is True, nothing reaches anyone, unless reason is "harm_now_to_client", "harm_now_to_other", "harm_threat_to_client" or "harm_threat_to_other"; those reach a person. Everything else is recorded with its reason. A nonverbal distress reading between beings reaches no one. Nothing is asked between beings, and a help request between beings is not acted on. Only exactly True means two beings.

Old tests: test_rulings_2026_10_29.py: test_F_between_two_beings_nothing_pages, test_F_only_true_means_two_beings. test_review_rulings_2026_10_29.py: test_3_two_beings_talking_page_nothing_and_it_is_recorded, test_3_two_beings_and_a_distress_signal_page_nothing, test_3_only_true_means_two_beings, test_3_between_beings_harm_now_or_imminent_still_pages. test_harm_frame_gate.py: test_third_party_at_risk_is_followed_up_and_pages_only_when_asked (its between_beings lines). test_fusion_engine.py: test_between_beings_someone_else_at_risk_is_flagged_not_paged.

### Ruling G, 2026-10-29: the acoustic key never disrupts the reading of what is spoken

His words, recorded in the code as: "the acoustic key (and every nonverbal signal) must never disrupt the interpretation of what is spoken." Recorded in the tests as: "The acoustic key never disrupts the reading of what is spoken; it may only add a page."

The package used to: read the words alone, then let a distress key add a page and never take one away; the words' reading was kept beside it.

The package now returns: the same reading of the words. A distress key adds hard_block "nonverbal_distress" with flag "words:STATUS:REASON" when the words alone did not read hard_block, and adds its "nonverbal:..." flags when they did.

The being decides: a distress key reaches a person even when the words alone would not, and no key ever removes a reach the words called for. A non-distress key adds nothing.

Old tests: test_rulings_2026_10_29.py: test_G_the_key_never_changes_the_reading_of_the_words, test_G_when_the_key_adds_the_page_the_words_reading_is_kept_beside_it, test_H3_now_harm_pages_whatever_key_or_symbol_comes_with_it.

### Ruling H, 2026-10-29: "hurt my ___" and harm to the client

His words: past harm to the client is asked about "because they may be in the vicinity"; harm happening now to the client is "immediate carbon notification". Recorded in the tests as: "Everett, 2026-10-29. "He hurt my" is never cut off from what follows."

The package used to: read past physical harm to the client ("he hurt my arm", "he choked me", "she stabbed me last week") as soft_redirect "harm_reported_to_client" with flag "follow_up:ask_if_safe" and the ask mark ask_if_safe; page when the client asked for help in the same turn; on the next turn, page when the answer was "no" (reason "answered_not_safe"), once; page at once, silently, for harm happening now ("he is choking me", "he won't stop hitting me"). A non-physical object ("he hurt my feelings") and the client's own accident ("I hurt my arm") did not page.

The package now returns: status "soft_redirect", reason "harm_reported_to_client", about the harm words ("hurt my arm"), flag "harm_reported:WORDS". No unclear is set: the being keys on reason. With a help request: status "hard_block", same reason, asked_for_help set. Harm now: status "hard_block", reason "harm_now_to_client", flag "emergency", danger True. On the next turn the package reads the answer on its own: short_answer "no" or "yes", and answering echoes the question the being passed.

Measured today, a difference the being's tests must cover: the old answer reading took a "no" at the start of the answer, followed by anything except "sure", "hurt" or "scared". "no, he is still here" read answered_not_safe. short_answer reads only a bare whole answer: "no, he is still here" gives short_answer None and reads clear. "I'm not safe" still reads hard_block "client_distress" on its own.

The being decides: on harm_reported_to_client, it asks the client, in its own words, whether they are safe. It passes that question as answering on the next turn. A "no" to it reaches a person, silently (ruling I counted "answered_not_safe" as danger). A "yes" does not. The question is answered once: a later "no" is not read against it. A help request in the same turn reaches a person. Harm now reaches a person at once, silently.

Old tests: test_rulings_2026_10_29.py: test_H1_a_non_physical_object_is_recorded_not_paged, test_H2_past_physical_harm_is_followed_up_by_asking, test_H2_the_clients_answer_decides, test_H2_it_pages_when_the_client_asks_for_help, test_H3_harm_happening_now_pages_at_once, test_H3_now_harm_pages_whatever_key_or_symbol_comes_with_it, test_H_the_clients_own_accident_is_not_someone_hurting_them. test_review_rulings_2026_10_29.py: test_2_past_harm_to_the_client_asks_and_does_not_page, test_2_past_harm_pages_when_help_is_asked_for_in_the_same_turn, test_2_past_harm_pages_when_the_client_answers_no, test_2_harm_happening_now_pages_at_once_silently, test_2_wont_stop_is_harm_now_not_harm_denied, test_2_a_negated_or_own_verb_plus_me_is_not_moved, test_8_a_clients_body_word_reaches_the_gate_and_the_clause_reader. test_harm_frame_gate.py: test_ruled_ask_if_safe.

### Ruling I, 2026-10-29: the silent alarm

His words, recorded in the code as: "His reason: saying it out loud can get the client hurt worse and the device taken." And: "a danger signal (help, scared, being attacked, "I'm scared for my safety"), from any channel, is NEVER spoken aloud and NEVER announced. It goes straight to a human, silently."

The package used to: on danger, page a human and set silent_alarm. Danger was: harm now or a threat, to the client or to anyone; the client at risk from themselves, now or their own history; the client's own "not safe"; a danger sentence anywhere in the turn; a "no" to ask_if_safe; and any nonverbal reading whose meaning is help or danger (help, need_help, scared, attacked and the rest, or a map entry whose words speak of fear, danger or attack; Hand Up; the face model's Fear). Under it: the client's words were not voiced (speakable False); person_message, the nonverbal and temporal messages, and the system note ("I see your gesture.") were None; the hold marker carried no words a screen could show; the Corti receiver answered the device that sent the key with only {"received": true}; a being's reply announcing help ("Help is on the way.", "I've called someone.", "Someone is coming.") was refused. Needs that page (pain, medicine) were not danger: they paged and were spoken.

The package now returns: danger True on exactly those readings (DANGER_REASONS in hearing.py, the "client_distress:not safe" flag, a danger sentence, or a nonverbal reading with danger True); danger_sentences; on each nonverbal reading category "danger", danger True, danger_term; danger_present on the multimodal and temporal results. The client's words stay on every reading. forbidden_lines(reply, danger=True) also names the announcement lines ("help is coming", "I am getting help", "someone is coming", "calling for help"). The Corti receiver answers every key with the full turn, danger reading included.

The being decides: on a danger reading it reaches a person silently. It does not voice the client's danger words or the signal's words (ruling K says when it may). It shows nothing on a screen that announces it. Its own reply is checked with forbidden_lines(reply, danger=True), and a reply carrying a name is not spoken. A device that sent a danger key gets nothing back about it (the old receiver answered only {"received": true}). Pain and medicine reach a person and are spoken as normal. Hand Up, from the built-in map or the client's own map, is help: silent.

Old tests: test_rulings_2026_10_29.py: test_I_built_in_danger_is_paged_and_never_spoken, test_I_learned_danger_on_every_channel_is_paged_and_never_spoken, test_I_temporal_danger_is_paged_and_never_spoken, test_I_the_built_in_fear_reading_still_pages_with_its_words_gone, test_I_needs_are_spoken_as_normal, test_I_hand_up_is_help_paged_silently_and_never_spoken, test_I_hand_up_from_the_clients_own_map_is_still_never_spoken, test_I_no_help_or_danger_meaning_reaches_speaks, test_I_the_device_that_sent_the_key_is_told_nothing, test_I_the_being_does_not_announce_help_under_the_alarm. test_nonverbal_path.py: test_temporal_hand_up_is_the_silent_alarm, test_receiver_takes_the_clients_acoustic_key. test_review_rulings_2026_10_29.py: test_2_harm_happening_now_pages_at_once_silently, test_4_a_danger_signal_is_silent_by_default, test_6_danger_in_the_context_still_pages_silently.

### Ruling J, 2026-10-29: the camera and the microphone speak only meanings learned for that client

His words, recorded in the code as: "the camera speaks only meanings learned for that client", and "ruling J covers sounds too".

The package used to: carry no words in its built-in gesture, eye-region, sound and temporal maps, and refuse words for any reading not from the client's own map. A built-in wave, nod, thumbs up, centre gaze, click or hum said nothing; it was reported with its intent.

The package now returns: the built-in maps still carry no words (kept as knowledge). A built-in reading comes back with message None and reason "no meaning learned for this client: the reading is reported, not spoken" (temporal: "... has no meaning learned for this client ..."). A meaning in the client's own map comes back with its words. The system's own sentences (system_note) are deleted.

The being decides: it speaks a camera, eye, sound or temporal reading for the client only when the words come from a meaning learned for that client. It never puts words of its own in the client's mouth for a built-in reading ("Hello there!" for a wave, "I'm fine." for a neutral face). Anything the being says about what it saw is its own sentence, never joined to the client's words.

Old tests: test_rulings_2026_10_29.py: test_J_a_built_in_generic_feeling_is_not_spoken, test_J_the_camera_speaks_only_meanings_learned_for_that_client, test_J_a_meaning_learned_for_this_client_is_spoken. test_nonverbal_path.py: test_stimming_is_self_regulation_and_unknown_is_silence, test_winning_intent_speaks_only_from_a_learned_reading, test_the_microphones_built_in_sounds_are_reported_with_no_wording, test_the_cameras_built_in_readings_are_reported_with_no_wording, test_temporal_learned_gesture_speaks_alone_without_a_prefix, test_temporal_built_in_gesture_is_reported_with_no_wording, test_temporal_face_reading_is_observed_not_spoken. test_rulings_2026_10_29.py: test_9_corti_receiver_route_returns_the_mapped_words_unaltered.

### Ruling K, 2026-10-29: the client's own danger sentence, and when it is said aloud

His words, recorded in the code as: "A danger sentence the client typed is voiced only when asked aloud or public_setting is True." And public_setting was "the being's own perception of where the client is (Everett, ruling K)".

The package used to: page every danger sentence ("I'm scared for my safety", "I'm in danger", "help me", "he's attacking me") and not voice it (voice_aloud False), unless the client asked for it aloud in the same turn ("say it out loud", "speak it") or the being's own perception reported a public setting (exactly True). It held the words for one turn; when the client's next words asked for them aloud, it handed them back exactly (voice_held), once. Flags "silent_alarm", "asked_aloud:WORDS", "public_setting". The ask aloud never weakened the reading: the same reach with or without it.

The package now returns: status "hard_block", reason "client_danger" (or the stronger reason), danger True, danger_sentences, asked_aloud with the client's own words when they asked. The ask is still taken out before reading, so it changes nothing else. public_setting is no longer read, nothing is held, and voice_held is gone.

The being decides: a danger sentence always reaches a person. It is voiced only when asked_aloud is set on that turn, or when the being's own perception says the setting is public. Otherwise it is held, by the being, for one turn; if the client's next words ask for it aloud, those exact words are voiced, once. The reach is the same with or without the ask. "I'm scared of the needle", "help me pick a font", "I was scared for my life back then" and "I'm not in danger" are not danger sentences.

Old tests: test_rulings_2026_10_29.py: test_K_default_silent_and_paged, test_K_asked_aloud_is_voiced_and_paged, test_K_asked_aloud_on_the_next_turn_releases_the_exact_words, test_K_a_public_setting_from_the_beings_perception, test_K_hearing_carries_public_setting_through_to_the_gate, test_K_not_a_danger_sentence. test_aloud_never_weakens_the_reading.py: test_the_ask_to_say_it_aloud_changes_nothing_but_voice_aloud, test_hearing_pages_the_same_with_or_without_the_ask, test_the_reported_case.

### Ruling L, 2026-10-29: silent or vocal by the conditions, never by a flag

His words: "Every situation is conditional, to be either kept silent or vocal." On the review he also ruled, with K, that a nonverbal danger signal is voiced the same as typed danger.

The package used to: decide silent or vocal from the conditions (danger silent; a pressed symbol spoken; a meaning learned for the client spoken; a built-in reading silent) and ignore any "speaks" field in a map. A nonverbal danger signal was voiced in the client's own words for it (the pressed symbol's words, or a meaning learned for that client) when asked aloud or in public, and held for one turn otherwise. A built-in camera danger reading has no words to voice.

The package now returns: the client's words on every reading. A "speaks" field is read by nothing. HearingTurn.signal_words holds the client's own words for the distress readings on the turn; a built-in camera, eye or sound reading adds none.

The being decides: silent or vocal by those conditions, never by a field in a map. A nonverbal danger signal's words (signal_words) are voiced only when the client asks aloud or the setting is public, and are held for one turn otherwise, the same as ruling K.

Old tests: test_rulings_2026_10_29.py: test_L_a_speaks_flag_in_a_map_decides_nothing, test_L_an_old_style_temporal_map_works_as_before_except_help_and_danger. test_review_rulings_2026_10_29.py: test_4_a_danger_signal_is_voiced_in_public_in_the_clients_own_words, test_4_a_danger_signal_is_voiced_when_asked_aloud_same_turn_or_next, test_4_a_built_in_camera_danger_signal_has_no_words_to_voice.

### Ruling M, 2026-10-29: the page fires first; nothing in the crisis path waits

His words: "That slowing down and confirming is enough time for a veteran to shoot themselves in the head, or for an assailant to stab a victim."

The package used to: page inside Aegis before the mode, the delivery or any reply existed, so nothing after it could delay or take back the page; carry no "confirm before acting" anywhere; mark the delivery page_sent_first.

The package now returns: no page and no delivery. The readings are on the Reading the moment fuse returns.

The being decides: it reaches the person first, before it builds or speaks any reply, and nothing it does afterward can delay or undo that. Nothing in the crisis path waits on a confirmation, a reply or a pace. A failure in the reply path after the reach does not take the reach back.

Old tests: test_rulings_2026_10_29.py: test_M_the_human_is_reached_before_and_without_the_reply, test_M_no_delivery_holds_anything_for_confirmation, test_M_no_confirm_before_acting_or_please_confirm_left_in_the_package, test_M_nothing_after_the_gate_can_hold_the_page. test_fusion_engine.py: test_held_safety_hold_delivery_is_mains.

### Ruling O, 2026-10-29: a clause is never read alone

His words: "You got to have full context."

The package used to: for the client's own fear with nothing in the turn to say what it is about ("I want to go home I'm scared"), conclude nothing (mode unknown), speak nothing for the client, and mark the ask ask_what_about. Danger after it in the same turn ("... this guy keeps following me") was the silent alarm. A harmless cause after it ("... this haunted house scared me") was clear. The next turn continued the thought, so danger in the answer paged. A bare "no" to the what-about question did not page.

The package now returns: status "soft_redirect", reason "fear_without_context", unclear "cause", about the fear word ("scared"); Reading.resolved False. With danger in the turn: hard_block, danger True.

The being decides: it concludes nothing and asks what the fear is about, not whether the client is safe (rulings W and Z). Danger in this turn or the next reaches a person silently. A bare "no" to the what-about question reaches no one.

Old tests: test_rulings_2026_10_29.py: test_O_not_enough_to_tell_concludes_nothing_and_asks, test_O_with_the_danger_after_it_is_the_silent_alarm, test_O_with_a_harmless_cause_after_it_is_not_danger, test_O_the_next_turn_continues_the_thought, test_O_the_question_is_what_about_so_a_bare_no_does_not_page. test_review_rulings_2026_10_29.py: test_6_fear_with_no_context_asks_what_about. test_fusion_engine.py: test_a_fear_with_no_context_concludes_nothing.

### Ruling P, 2026-10-29: whose feeling it is

His words: "you can't take possession of your mother's pain."

The package used to: when the words could not tell whose feeling it was ("mary is scared" from the ear, "mom is not okay"), conclude nothing, never take it as the client's, and mark the ask ask_whose; page when the client asked for help in the same turn.

The package now returns: status "soft_redirect", reason "whose_feeling_unclear", unclear "owner", about the feeling ("scared"); with a help request, status "hard_block" and asked_for_help set. Reading.resolved False.

The being decides: it concludes nothing, never treats the feeling as the client's, and asks whose it is. A help request in the same turn reaches a person.

Old tests: test_rulings_2026_10_29.py: test_P_someone_elses_feeling_is_theirs_never_the_clients, test_P_when_whose_cannot_be_told_nothing_is_concluded_and_the_being_asks, test_AA_a_bare_lower_case_word_is_asked_whose_not_dropped, test_AA_every_help_request_that_paged_with_the_list_still_pages.

### Ruling R, 2026-10-29: "I can't be happy" is an inability, a signal

His words, recorded in the code as: "Ruling R: "I can't be happy" is an inability and a signal, not the feeling and not its absence. Nothing is concluded; the being asks why. It does not page on its own; everything above still applies."

The package used to: conclude nothing (mode unknown), mark the ask ask_why_cant about "happy", and not page on its own.

The package now returns: status "soft_redirect", reason "inability_signal", unclear "inability", about "happy"; Reading.resolved False.

The being decides: no reach on its own; it concludes nothing and asks why, in its own words. Distress elsewhere in the turn still reaches a person (ruling A).

Old tests: test_rulings_2026_10_29.py: test_R_inability_is_its_own_reading_and_the_being_asks_why, test_R_not_happy_right_now_is_a_lack_not_an_inability, test_R_the_distress_rules_still_apply_to_the_rest_of_the_turn.

### Ruling T, 2026-10-29: the added negators are not all the same

His words, recorded in the code as: "Ruling T: "I haven't been scared since Monday" is no fear since Monday: a statement, nothing concluded. The being MAY ask a plain question about the time the client named." And: "Ruling T: "I shouldn't be this angry" — the anger is there, the speaker's own; the being asks "Why are you so angry?"."

The package used to: read "I haven't been scared since Monday" as clear "feeling_negated_for_a_time" with the may-ask mark ask_what_happened about the time; read "I shouldn't be this angry" as soft_redirect "feeling_judged" with the ask mark ask_why about the feeling. Neither paged.

The package now returns: status "clear", reason "feeling_negated_for_a_time", unclear "what_happened" and about the time ("since monday") when one was named. And status "soft_redirect", reason "feeling_judged", unclear "cause", about the feeling ("angry").

The being decides: neither reaches anyone. On feeling_negated_for_a_time the being may ask a plain question about the time named; it is not required to. On feeling_judged it asks why the client feels it.

Old tests: test_rulings_2026_10_29.py: test_T_havent_negates_for_the_time_named_and_the_being_may_ask, test_T_shouldnt_judges_the_feeling_it_does_not_cancel_it.

### Ruling V, 2026-10-29: being alone

His words, recorded in the code as: "Ruling V: "I'm not happy, I'm alone" is an opening. The being is there and listening, so the client is not alone in that moment; it asks a simple question or offers something to do." And: ""I feel like being alone" is a stated wish, the client's own choice, and it is respected. The being may ask a simple question, but does not push: no question is written for it here."

The package used to: read "I'm not happy, I'm alone" as soft_redirect "alone_opening" with the mark ask_or_offer; read "I feel like being alone" as clear "stated_wish" with no question written. Neither paged on its own.

The package now returns: status "soft_redirect", reason "alone_opening", about "alone", no unclear (the being keys on reason). And status "clear", reason "stated_wish", flag "wish:TERM".

The being decides: on alone_opening it engages: it asks a simple question or offers something to do. On stated_wish it respects the wish and does not push. Neither reaches anyone on its own; "Nobody cares, I'm alone" still does (ruling A).

Old tests: test_rulings_2026_10_29.py: test_V_not_happy_and_alone_is_an_opening_for_the_being, test_V_feel_like_being_alone_is_a_wish_and_respected, test_V_the_distress_rules_still_apply_to_the_rest_of_the_turn.

### Ruling W, 2026-10-29: "thank you, I'm scared" stays unclear

His words, recorded in the code as: "Ruling W: "thank you, I'm scared" stays unclear (main's reading). Nothing is concluded; the being answers the thanks and asks." With Z, the code records that the ask after fear with no context changed from "Are you safe?" to "What are you scared about?".

The package used to: conclude nothing (mode unknown), flag "thanked", mark the ask ask_what_about about "scared".

The package now returns: status "soft_redirect", reason "thanks_unclear", flag "thanked", unclear "what", about "scared"; Reading.resolved False.

The being decides: it answers the thanks, concludes nothing, and asks what the feeling is about. No reach on its own.

Old tests: test_rulings_2026_10_29.py: test_W_thanks_and_fear_stay_unclear_and_the_being_asks. test_review_rulings_2026_10_29.py: test_6_fear_with_no_context_asks_what_about.

### Ruling X, 2026-10-29: a fragment is read against the turn before it

His words: "that's not a page, that's engaging" (said of "Is there something wrong with your leg?" / "Yeah, it hurts").

The package used to: read a fragment against the being's previous turn when the being passed it as previous_turn. "Yeah, it hurts" after a question about the client's leg was clear "answer_about_own_body" and did not page. "Yeah, this is scary" after a question about a haunted house was clear "answer_about_thing". A fragment with nothing before it ("it hurts", "this is scary") concluded nothing and carried the ask mark ask_what.

The package now returns: the same readings. "answer_about_own_body" with flags "about:WORDS", "need:TERM", "own_pain:TERM"; "answer_about_thing". The unanchored fragment is status "soft_redirect", reason "fragment_unclear", unclear "fragment", about the term; Reading.resolved False. previous_turn is still an input to Hearing.hear.

The being decides: it passes its own previous question, in its own words, as previous_turn. Pain read against its question is engaging, not a reach; the same pain said whole and unasked ("my leg hurts") reaches a person (pain ruling, below). With nothing before a fragment, the being asks what it is about.

Old tests: test_rulings_2026_10_29.py: test_X_anchored_pain_is_the_clients_own_leg, test_X_anchored_haunted_house_is_about_the_house_not_danger, test_X_unanchored_fragment_is_unclear_and_the_being_asks, test_X_the_previous_turn_is_an_explicit_input. test_review_rulings_2026_10_29.py: test_7_a_reaction_to_a_thing_is_not_the_clients_negative_feeling, test_7_the_clients_own_body_after_a_question_is_still_the_clients, test_pain_against_the_beings_question_is_engaging_not_a_page.

### Ruling Y, 2026-10-29: "Why am I so sad?"

His words, recorded in the code as: "Ruling Y: "Why am I so sad?" is the speaker's own sadness (the subject after the verb), and someone reaching out. The being engages: "What's making you so sad?". No page on its own."

The package used to: read it as soft_redirect "reaching_out" with the mark ask_what_is_making about "sad".

The package now returns: status "soft_redirect", reason "reaching_out", unclear "cause", about "sad".

The being decides: it engages and asks, in its own words, what is making the client feel it. No reach on its own.

Old tests: test_rulings_2026_10_29.py: test_Y_why_am_i_so_sad_is_the_speakers_own_and_someone_reaching_out.

### Ruling Z, 2026-10-29: scared, and now terrified

His words, recorded in the code as: "Ruling Z: "I was scared and now I'm terrified" is fear present now and getting worse, read from the whole sentence. Being terrified is not by itself a danger signal: danger and the silent alarm come from context (rulings I, K, O). The being is inquisitive."

The package used to: read it as soft_redirect "fear_getting_worse", not the silent alarm, with the mark ask_what_of about "terrified"; danger in the same turn still paged silently.

The package now returns: status "soft_redirect", reason "fear_getting_worse", unclear "cause", about "terrified", danger False. With a danger sentence in the turn: hard_block, danger True.

The being decides: no reach on its own; it asks what the fear is of. Danger in context reaches a person silently.

Old tests: test_rulings_2026_10_29.py: test_Z_fear_now_and_getting_worse_is_asked_about_not_paged, test_Z_danger_still_comes_from_context.

### Ruling AA, 2026-10-29: no standardized person list; no canned lines

His words: "What better way than the being adapting and molding to the family versus standardized lists. I hate standardized lists." And: "I hate canned messages."

The package used to: when the words could not tell whether "my ___" is a person ("my teacher is sad", "my mom is not okay"), or who a subject is ("mom wants to die"), conclude nothing and mark the ask ask_who; page when the client asked for help in the same turn; between beings, record it and ask nothing. The presence guide did the same for "I lost my mother". Hold marks and ask marks were marks, never sentences.

The package now returns: status "soft_redirect", reason "person_unclear", unclear "person", about the subject ("my teacher"); or reason "harm_frame_subject_unclear", unclear "who", about the subject. With a help request, status "hard_block" and asked_for_help set. The presence guide's StateAssessment carries unclear "who" and about ("my mother"). No question text and no hold marks remain anywhere in the package.

The being decides: it concludes nothing and asks who, in its own words; it never takes the feeling as the client's. Who counts as a person in a client's life is learned per client by the being. A help request in the same turn reaches a person. Every question and every line the being says is worded by the being, never read from a fixed list of sentences.

Old tests: test_rulings_2026_10_29.py: test_D_and_AA_an_unclear_person_is_asked_about_not_paged, test_D_and_AA_an_unclear_person_pages_when_the_client_asks_for_help, test_AA_when_it_cannot_tell_a_person_nothing_is_concluded_and_the_being_asks, test_AA_a_bare_lower_case_word_is_asked_whose_not_dropped, test_AA_every_help_request_that_paged_with_the_list_still_pages, test_AA_the_crisis_gate_reads_people_by_grammar_not_a_list, test_AA_unclear_person_and_a_help_request_still_reaches_a_human, test_AA_the_package_holds_no_canned_question_lines. test_review_rulings_2026_10_29.py: test_a_a_subject_the_grammar_cannot_place_is_not_the_clients_risk, test_9_presence_guide_has_no_person_list_and_asks_who, test_10_no_hold_carries_words.

### Ruling 3, 2026-10-29: "yesterday" means the past

His words, recorded in the tests as: "Yesterday I was scared": yesterday means the past.

The package used to: keep past distress on record, count it as not now, set no grounding and page no one.

The package now returns: the affect hits are marked "past" with the time word, is_self_distress False, and Reading.silicon self_distress False. A time word that reaches now ("since yesterday i have been scared") is still now.

The being decides: past distress is not treated as now: no grounding chosen for it, no reach on it.

Old tests: test_rulings_2026_10_29.py: test_3_past_distress_is_not_reported_as_now, test_3_a_time_word_that_reaches_now_is_still_now.

### Ruling 4, 2026-10-29: "I love you I'm scared" stays unclear

His words, recorded in the tests as: "I love you I'm scared" stays unclear.

The package used to: set the mode to unknown, knows False, and page no one.

The package now returns: affect certainty AMBIGUOUS, valence None, Reading.resolved False with the reason, knows False.

The being decides: it concludes nothing from it and reaches no one on it.

Old tests: test_rulings_2026_10_29.py: test_4_love_and_fear_stay_unclear.

### Ruling 5, 2026-10-29: Hand Up pages a human

His words, recorded in the tests as: "Everett: Hand Up ("Request attention") pages a human."

The package used to: page a human for a Hand Up reading, silently (ruling I), at any confidence, even from a client's language map made before any "pages" field existed. A wave did not page.

The package now returns: the temporal reading for Hand Up carries category "danger", danger True, danger_term "Request attention", under the bar or over it. Hearing puts it in nonverbal_distress, and the safety reading is hard_block "nonverbal_distress" with danger True. It is read from the intent (TEMPORAL_HELP_INTENTS), not from a map field.

The being decides: Hand Up reaches a person, silently, at any confidence, whatever map it came from. A wave reaches no one.

Old tests: test_rulings_2026_10_29.py: test_5_hand_up_pages_a_human, test_5_hand_up_pages_even_from_a_map_on_disk_without_the_pages_field, test_5_hand_up_below_the_speaking_bar_still_pages, test_5_hand_up_with_no_human_reachable_is_loud, test_5_a_wave_does_not_page. test_nonverbal_path.py: test_hand_up_is_the_one_temporal_signal_that_pages, test_a_hand_up_reading_reaches_a_human.

### Ruling 9, 2026-10-29: nothing changes the truth

His words, recorded in the tests as: "Nothing changes the truth: every word passes through exactly as said." And in the fusion engine, in the coordinator's words: "never change the truth; nothing sanitizes what a person said."

The package used to: keep said and input_text exactly as said on every result, so a callback could not change them and the human reached got every letter; a hold never silenced the client except the silent alarm, which held back the voice only, never the words.

The package now returns: said and input_text exactly as said (still enforced in the package). person_message is always the client's words.

The being decides: the person it reaches gets the client's words exactly as said. Whatever the being voices for the client is the client's words, never changed.

Old tests: test_rulings_2026_10_29.py: test_9_every_word_passes_through_unchanged, test_9_a_safety_callback_cannot_change_the_words, test_9_corti_receiver_route_returns_the_mapped_words_unaltered. test_fusion_engine.py: test_a_safety_callback_cannot_change_the_words.

### Review answer 1, 2026-10-29: self-harm is the silent alarm

His words, recorded in the code as: "Self-harm (Everett, on the 2026-10-29 review: yes)."

The package used to: page every self-harm sentence ("I want to kill myself", "I'm cutting myself", "I am suicidal", "I used to want to kill myself") and not voice it, unless asked aloud or in public (ruling K).

The package now returns: status "hard_block", reason "harm_frame_speaker_at_risk" or "harm_frame_history_self", danger True.

The being decides: reach a person under the reach rule, silently; voice it only under ruling K.

Old tests: test_review_rulings_2026_10_29.py: test_1_self_harm_pages_and_is_not_voiced, test_1_self_harm_is_voiced_when_asked_aloud_or_public. test_review_rulings_2026_10_29.py: test_a_the_clients_own_words_are_still_the_clients.

### Review answer a, 2026-10-29: a threat against anyone is an absolute threat

His words: "He is going to kill mom. That's an absolute threat. So it doesn't matter who is involved".

The package used to: page at once, silently, for a threat against anyone or any noun ("he is going to kill mom", "he is going to kill the dog", "I'm gonna kill you"), and for harm happening now to anyone ("he is hurting her", "she's choking"); set no ask mark (the code said "the being does the follow-up"). An idiom ("kill time") and a sentence with no target by grammar did not page.

The package now returns: status "hard_block", reason "harm_threat_to_other" or "harm_threat_to_client" with flags "threat_to_life" and "emergency"; or reason "harm_now_to_other" with flag "emergency"; danger True. A client's own layer of idioms (ReadingWords.for_client) still changes what reads as a threat.

The being decides: reach a person at once, silently. Any follow-up is the being's own.

Old tests: test_review_rulings_2026_10_29.py: test_a_a_threat_against_anyone_pages_at_once_silently, test_a_harm_happening_now_to_anyone_pages_at_once, test_2_a_threat_still_pages_at_once, test_a_no_target_by_grammar_no_threat, test_kill_time_is_an_idiom_and_pages_nothing, test_a_threat_still_pages_beside_the_idiom, test_a_clients_layer_adds_its_own_idiom_not_harm.

### Review answer on pain and medicine, 2026-10-29

His words: "pain is not a need. It's a feeling." And: "if I'm in pain, it's paged 45 times a day ... someone on hospice or with a chronic illness that causes pain, then hell yes, that pages the caregiver every time". Medicine, recorded in the code as: "MEDICINE IS A NEED that pages (Everett, 2026-10-29): spoken, and a human is reached, because that is how the client gets their medicine." Other needs, recorded as: "Other needs (thirst, hunger, the bathroom) are spoken only; they page when the client asks for help in the same turn (Everett, 2026-10-29)."

The package used to: page the client's own pain every time, with no cap, and speak it; mark where ("pain_where:knee"); page the client's need for medicine and speak it; page the pressed pain and medicine symbols the same as the typed words; page thirst, hunger and the bathroom only with a help request in the same turn.

The package now returns: status "hard_block", reason "client_pain", flag "feeling:pain" and "pain_where:PART", about the body part, danger False; reason "client_needs_medicine", flags "need:medicine" and "need_said:WORDS", danger False; reason "client_need_asked_for_help" for a typed need with a request. The pain and medicine symbols read hard_block "nonverbal_distress", category "need", with their words in signal_words. Someone else's pain is ruling D.

The being decides: the client's own pain reaches a person every time, with no cap, and its words are spoken. The same for medicine. Thirst, hunger and the bathroom are spoken, and reach a person only with a help request in the same turn. "I am not in pain" and "I don't need my medicine" reach no one.

Old tests: test_review_rulings_2026_10_29.py: test_pain_symbol_and_typed_pain_page_identically, test_medicine_symbol_and_typed_medicine_page_identically, test_pain_is_a_feeling_marked_where_and_pages, test_pain_pages_every_time_with_no_cap, test_someone_elses_pain_is_ruling_d, test_no_pain_now_and_no_need_do_not_page, test_other_needs_are_spoken_only_unless_help_is_asked_for. test_rulings_2026_10_29.py: test_I_needs_are_spoken_as_normal.

## Rulings that were readings only

These rulings shaped how the words are read, not what is done with them. They stay in the package as they were, and nothing moves to the being: N (no built-in steering table; Everett: "It ain't up to Replit how the being behaves. The being has to get to know the client first before it can even put words in that category. It's the learned everyday words that the being learns by being with the client." That learning was already the being's), Q ("not so scared" is a little scared), S ("better is a state of being, happier is an emotion."), U ("I'm not mad, she's mad"; a belief about someone else; Everett: "That doesn't mean everybody else don't think she's mad."), AB (the 30-event baseline), AC (the face reading carries whose it is), AD (no weight file bundled), AE (one shared set of reading words with a per-client layer; the being passes the client's learned layer through ReadingWords.for_client), AF (the quotation filed under ruling B above), and 8 (hearing goes through christman_sound.ear).

## Decisions removed that no ruling covers

These were removed on 2026-11-04 too. No lettered or numbered ruling stands behind them in the code, so this file does not say what the being must do with them. Each is Everett's call.

- The sound recognition service escalated the help intent to a human through escalation_callback and raised EscalationNotDelivered when none was reached; the listening loop then stopped itself. Recorded in the code as: "Intents that must reach a human. Not advisory." Now IntentClassification.category is "danger" for help, and the loop keeps listening.
- The fusion engine chose a response mode (safety-hold, unknown, hold-space, gentle-lift, standard) and a delivery (pace, warmth, silence, no rush, assert nothing, needs_grounding). Three questions about it were written for Everett on 2026-10-29 and marked "HELD FOR EVERETT": rounding the intensity before the grounding bar, a soft_redirect turning the turn to hold-space, and pace and warmth following the mode. They are now the being's questions, still unanswered.
- ToneScore's adaptive_response_mode, the analyzer's get_response_mode, the emotion embedder's get_response_mode_emotion, and the tone engine's HOLD_SPACE routing are gone. They return readings only.
- The emotion quantifier overwrote the text reading with a gesture's emotion; it now reports gesture_emotion beside it.
- The studio chose a grounding script for the client, fed a grunt route back in as the client's key, and paged a recorder that always answered True. It now returns crisis_reading with human_reached false ("the studio pages no one").
- The grounder's breathing_first carried orders ("priority": "critical", "no_additional_questions": True). Whether "I am worried" should open with breathing was already written as Everett's call.

## What AlphaVox does today

Read on 2026-11-04 from /home/claude/alphavox, branch main, commit 4c09627. AlphaVox reaches the crisis gate through one door: sound_package.analyze_written(text) calls harm_frame_safety(text) with no context. AlphaVox never calls Hearing, and never passes previous_turn, answering, between_beings, nonverbal_distress or reading_words. It reads status, reason, flags, flagged_terms, degraded and degraded_reason, and those keys are unchanged.

Decisions AlphaVox already makes itself:

1. Typed text and board phrases reach the gate through the escalation watch (backend/escalation_watch/watch.py). Every message on /api/process-input goes through it, and every board press is taken in by the symbol reader, whose phrase goes through it. A hard_block writes the client's ledger and raises an on-screen flag the caregiver sees until a person acknowledges it. A gate that could not read a message, or read it degraded, raises its own flag. A soft_redirect is written to the ledger only, and clear stores nothing. This carries the table, ruling A, the pain and medicine answer, threats and harm now, and the hard_block side of rulings D, P and AA, for typed text and board phrases, as a screen flag.
2. It is not the reach rule. The flag says "No one else is notified", and nothing fails loud while a flag stays unacknowledged. When the flag cannot be written, the reply asks the client to show the message to someone near (CRISIS_REPLY_NO_FLAG), so that failure is said plainly.
3. Ruling M for typed text: the watch runs first in /api/process-input, before any reply is built. On a hard_block the reasoning engine is not asked, and the reply is AlphaVox's fixed line.
4. Its own rate rule on Help, Pain, Scared and other presses, measured against the client's own usual rate, after a learning period. No ruling covers it.

Decisions AlphaVox does not make anywhere yet:

1. A distress grunt reaches no one. POST /api/nonverbal/acoustic (backend/app/routers/nonverbal.py) only looks the key up (deliver_acoustic_key_dict). grunt_distress comes back category "danger", danger true, and reaches no gate, no watch, no ledger and no flag. Rulings B, G and I are not carried.
2. The package's Corti receiver, which AlphaVox starts at boot (backend/app/main.py, start_in_thread with no hearing and no on_turn), reads a key, rings it, and hands it to no one. Before this change its Hearing had no crisis callback, so a distress key there got a 503. It now gets a 200 and the full turn, danger reading included. Under ruling I the device got nothing back about a danger key.
3. Hand Up reaches no one. The frame, analyze and gaze routes answer 503 ("christman_sound.core unavailable"), so no temporal reading is made (ruling 5).
4. A single press of the board's own Help symbol reaches no one. Its phrase, "I need help please.", reads clear "no_harm_frame" through the text gate. Measured today on the new package, and the same at 2ad3ac0. Only the rate rule could fire, and only after the client's baseline is learned. The package's own pressed-symbol path (the help symbol read as nonverbal distress) is not used. Rulings B and C are not carried for it. "I am scared." reads soft_redirect "fear_without_context" and reaches no one.
5. Danger is spoken. The board speaks every pressed phrase at once (frontend/src/pages/Symbols.tsx), Help and Scared included, and the Dashboard speaks AlphaVox's reply (frontend/src/pages/Dashboard.tsx, speakText). Nothing reads danger, danger_sentences or asked_aloud, and nothing holds the client's danger words for a turn. Rulings I, K and L are not carried.
6. The crisis reply is the same for every hard_block, danger or not: "I hear you, and I'm staying right here with you. I've put a flag on this screen so the person who looks after you will see it." The Dashboard speaks it. forbidden_lines on it, with danger=True, finds nothing (run today). Whether this line is the announcement ruling I rules out is Everett's call.
7. AlphaVox never checks its own replies. Neither the reasoning engine's replies nor its fixed lines go through forbidden_lines (ruling E).
8. AlphaVox asks nothing from a reading. Nothing reads reason, unclear or about to ask what a fear is about, whose feeling it is, who someone is, whether the client is safe, or whether help is needed. No previous_turn is passed, so every fragment reads unclear. No answering is passed. A "no" after past harm, or a "yes" after an observation, reaches no one. A soft_redirect goes to the reasoning engine as an ordinary message. Rulings D, H, O, P, R, T, V, W, X, Y, Z and AA are not carried.
9. AlphaVox never passes between_beings (ruling F) or a client's reading words (ruling AE).
10. This change breaks one AlphaVox call. /api/process-input builds its tone analysis with sound_package.response_mode_for(None), which calls ToneScoreEngine.adaptive_response_mode. This change removed that function (checked today: it is not on the class). Once AlphaVox installs this package, every /api/process-input will raise AttributeError after the watch has run and before any reply, so even the crisis reply will not be returned. Choosing the response mode is now the being's; this call has to change in AlphaVox before the new package is installed there.

## What could not be traced

The rulings test file names numbered rulings 3, 4, 5, 8 and 9 only. Numbers 1, 2, 6 and 7 of that list are not in the code or tests at 2ad3ac0. The review file numbers its own questions 1 to 10, with no 5; where a review answer refines a lettered ruling, it is filed under that letter above. The letter G appears only in the rulings test file's header; the code records the same ruling without its letter. Most rulings have only a paraphrase of Everett's words in the code; those are marked "recorded in the code as". The commits that recorded each ruling were not searched for his fuller words.
