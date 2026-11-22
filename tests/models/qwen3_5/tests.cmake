ninfer_add_test(ninfer_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_dspine_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspine_loading.cpp"
  LIBRARIES ninfer_model_loading)

# The DSpine load path (config parse + binding + tensor inventory) on a real V3-tailored
# artifact. It reads the manifest only - no payload, no device memory - so it is serial-labelled
# "real" rather than "gpu", and skips without NINFER_TEST_ARTIFACT.
ninfer_add_test(ninfer_qwen3_5_dspine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspine_real.cpp"
  LIBRARIES ninfer_model_loading)
set_tests_properties(ninfer_qwen3_5_dspine_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "real;dspine")

# The DSpine draft forward against a real V3 drafter artifact, on the device: the load footprint,
# the linear geometries the proposal path names, the proposal steps walked on the real weights,
# the supported bridge chain, and the sampled tail composed onto the target head. It materializes
# ~4 GiB on one card, so it is opt-in by NINFER_TEST_ARTIFACT and NINFER_TEST_DEVICE (D-29: a
# device probe never guesses a card) and skips, exit 77, without either.
# The accept-side row of the round (`serial_tail_accept_q_error`, the contract the drafts the
# serial tail publishes must satisfy before `speculative_accept_sparse_drafts` will take them) is
# the engine's own host code, compiled in as the tail's own unit test compiles it.
# P2 stage 4: the DSpine decode + accept seam, on the device and on a real V3-tailored artifact.
# It drives the new execution::dspine_propose_tail (the drafter's own head, tg/tsv tables and
# final norm) and hands its published drafts, candidate_ids and proposal_q to the engine's ONE
# sparse verifier - the call program/decision.cpp:84-93 makes for DFlash2 - over N>=8 rounds,
# with a planted argmax prefix per round and two teeth arms. It reports accepted tokens per
# round, the number D-40 put the project's bar at (DFlash2 K=7: 2.998). The target arm is analytic
# and says so in its own line: a real 27 B tower and a DSpine drafter do not coexist on one card
# here (D-42), so the V3 A/B is the a5000-2 lane's. Links the execution target because the seam
# under test is execution code; the accept-side contract checker is the engine's host code,
# compiled in as the tail's own unit test compiles it. Skips, exit 77, without the two envs.
ninfer_add_test(ninfer_qwen3_5_dspine_accept_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspine_accept_real.cpp"
          "${PROJECT_SOURCE_DIR}/src/models/common/dspine/dspine_serial_tail.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops ninfer_model_execution)
set_tests_properties(ninfer_qwen3_5_dspine_accept_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "real;dspine;gpu" TIMEOUT 900)

ninfer_add_test(ninfer_qwen3_5_dspine_forward_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspine_forward_real.cpp"
          "${PROJECT_SOURCE_DIR}/src/models/common/dspine/dspine_serial_tail.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_qwen3_5_dspine_forward_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "real;dspine;gpu" TIMEOUT 1800)

# The DSpine draft forward's linear gate, read out of the compiled shape tables at the geometries
# the real drafter declares (the shape half of the forward probe, wt-dspine-fwd @ 59830140). A
# host-side lookup, so it allocates nothing; it is labelled "gpu" only because it names the card it
# ran on, and it needs NINFER_TEST_ARTIFACT + NINFER_TEST_DEVICE (D-29: no device is guessed).
ninfer_add_test(ninfer_qwen3_5_dspine_shapes_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspine_shapes_real.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_qwen3_5_dspine_shapes_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "real;dspine;gpu" TIMEOUT 600)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_native_transactions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_native_transactions.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_mtp_history_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_history_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)
ninfer_test_interpose(ninfer_qwen3_5_prefix_real_test cudaEventQuery)

ninfer_add_test(ninfer_qwen3_5_prefix_admin_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_admin_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

add_test(NAME ninfer_qwen3_5_terminal_publication_real_test
  COMMAND ninfer_qwen3_5_prefix_real_test)
set_tests_properties(ninfer_qwen3_5_terminal_publication_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=terminal-publication")

ninfer_add_test(ninfer_qwen3_5_preemption_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_preemption_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_fairness_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_fairness_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_grammar_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_grammar_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_tools_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tools_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

add_test(NAME ninfer_qwen3_5_moe_offload_real_test
  COMMAND ninfer_qwen3_5_moe_real_test --offload)
set_tests_properties(ninfer_qwen3_5_moe_offload_real_test PROPERTIES
  ENVIRONMENT "NINFER_VRAM_BUDGET_MIB=16384")

ninfer_add_test(ninfer_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

add_test(NAME ninfer_qwen3_5_agent_continuation_real_test
  COMMAND ninfer_qwen3_5_prefix_real_test)
set_tests_properties(ninfer_qwen3_5_agent_continuation_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=agent-continuation")

# A real-model test owns the single GPU while its artifact is resident.
set(ninfer_qwen3_5_real_tests
  ninfer_qwen3_5_loading_real_test
  ninfer_qwen3_5_native_transactions_test
  ninfer_qwen3_5_mtp_history_real_test
  ninfer_qwen3_5_prefix_real_test
  ninfer_qwen3_5_prefix_admin_real_test
  ninfer_qwen3_5_terminal_publication_real_test
  ninfer_qwen3_5_agent_continuation_real_test
  ninfer_qwen3_5_preemption_real_test
  ninfer_qwen3_5_fairness_real_test
  ninfer_qwen3_5_grammar_real_test
  ninfer_qwen3_5_tools_real_test
  ninfer_qwen3_5_score_real_test
  ninfer_qwen3_5_vision_workspace_test
  ninfer_qwen3_5_dflash2_real_test
  ninfer_qwen3_5_dflash_prefill_real_test
  ninfer_qwen3_5_moe_real_test
  ninfer_qwen3_5_moe_offload_real_test
  ninfer_qwen3_5_dflash_real_test)
set_tests_properties(${ninfer_qwen3_5_real_tests} PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real")

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

# The DSpine proposal-tail COST curve (D-43's perf question, the criterion ⑤ denominator): the
# cost per round of linear_topk + dspine_serial_scores + dspine_serial_tail at K = 1, 3, 7, 15 on
# the real head/tg/tsv, CUDA-event timed, with the measured scaling exponent against the K^2
# reading and the break-even acceptance the wide window has to buy. It is a benchmark, and it is
# registered as a test because the probe that carries the tail's cost arms has to initialise CUDA
# the way the forward probe does: the same binary in the bench/ closure stops at the device bind
# with cudaErrorMemoryAllocation (and under lazy module loading it stops silently), while this
# closure reaches the device in 0.29 s. Its device setup is a verbatim copy of the forward probe
# below - the same Probe struct, the same materialization set, the same sources and libraries -
# because that is the probe family that runs this artifact (D-45: the #70 sibling stops before its
# first line, so the working path is the only accepted setup). Opt-in by NINFER_TEST_ARTIFACT +
# NINFER_TEST_DEVICE (D-29), skips exit 77, and it is RUN_SERIAL because it is a timing run.
ninfer_add_test(ninfer_qwen3_5_dspine_tail_cost_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspine_tail_cost.cpp"
          "${PROJECT_SOURCE_DIR}/src/models/common/dspine/dspine_serial_tail.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops)
set_tests_properties(ninfer_qwen3_5_dspine_tail_cost_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "real;dspine;gpu" TIMEOUT 3600)

# The DSpine PORT-FIDELITY probe (D-49's open item): the executor's own
# execution::dspine_append_context -> dspine_walk -> dspine_propose_tail, driven over
# parameters.dspine on a real V3 drafter for N >= 32 rounds, compared per published stream against
# the forward probe's walk (run_forward/run_bridge) on identical inputs. It is the probe that
# turns D-49's STRUCTURAL grade into a bit-parity table, and it is the first execution of
# dspine_walk: no probe drove it, and the decode loop that does needs a target tower. The load is
# the engine's own (read_definition -> describe_model -> resolve_load -> materialize_model ->
# execution::Parameters) so the BINDING is under test too - the packed Q/K/V parents and the
# stacked bridge slices that a probe-side over_object also computes - and one shard serves both
# arms, because this artifact's tower plus drafter is 24.115 GiB and the card's free memory is
# ~25.4 GiB: the probe refuses on the card's own numbers before allocating, rather than repeat
# D-47's stop. Its device bring-up is test_dspine_tail_cost.cpp's verbatim (D-47). Opt-in by
# NINFER_TEST_ARTIFACT + NINFER_TEST_DEVICE (D-29), skips exit 77; the tower load makes it slow to
# open, and it is RUN_SERIAL because it owns the card.
ninfer_add_test(ninfer_qwen3_5_dspine_exec_fidelity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspine_exec_fidelity.cpp"
          "${PROJECT_SOURCE_DIR}/src/models/common/dspine/dspine_serial_tail.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops ninfer_model_execution)
set_tests_properties(ninfer_qwen3_5_dspine_exec_fidelity_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "real;dspine;gpu" TIMEOUT 1800)

# The FIRST ACCEPTED/ROUND measurement for DSpine, against a REAL target (criterion 5). D-42 moved
# the number to the a5000-2 lane on the premise that the drafter cannot co-reside with the target
# tower on GPU 1; D-51 disproved it - the fidelity probe loaded the executor's own shard (tower AND
# drafter, 20.43905 GiB) on that card and walked 32 rounds bit-exact. This probe runs the round on
# `ninfer::Engine` (--spec dspine, K=15, batch 1, eager: graph_bodies.cpp:19 refuses a DSpine graph
# profile), so the four parts of the round are the engine's own: the target forward that produces
# the verified logits, run_dspine_draft's walk + tail, the ONE verifier
# ops::speculative_accept_sparse_drafts (decision.cpp:84-99, the call DFlash2 makes), and the
# accounting at session/commit.cpp:294-308 that turns its counts into stats.speculative. It reports
# accepted drafts/round on the prod policy (T=0.9, top_k 20, top_p 1.0, min_p 0.05) and a T=0 arm,
# on prod's own definition (+1 for the verified target token) so the row sits next to DFlash2's
# measured 2.998 (D-40) and the 3.250 break-even (D-48), plus the per-round histogram and the
# p0..p14 decay out of the depth census, and ms/round. The accepted column is prod's arithmetic; the
# card is GPU 1 with basalt live, so the timing columns are indicative, not the lane's. Opt-in by
# NINFER_TEST_ARTIFACT + NINFER_TEST_DEVICE (D-29: a device probe does not guess a card), skips exit
# 77, RUN_SERIAL because it owns the card and it is a measurement run.
ninfer_add_test(ninfer_qwen3_5_dspine_ab_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspine_ab_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)
set_tests_properties(ninfer_qwen3_5_dspine_ab_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "real;dspine;gpu" TIMEOUT 3600)

# H1 of the D-53 context-feed diagnostic: IS THE DSpine DRAFT PATH ACTUALLY FED THE TARGET'S STATE.
# D-53's lower bound (DSpine 0.85 accepted/round, and -74 % on a 316-token prompt against DFlash2's
# -15 %) names the drafter's context feed as the first suspect, and the three arms that make the
# measurement run at all (program/execution.cpp prefill, program/draft_commit.cpp commit,
# program/planning/startup.cpp local-state) are new and unreviewed. This probe drives the
# executor's own `dspine_append_context` -> `dspine_walk` -> `dspine_propose_tail` over the
# engine's own `PagedKVCache` and (a) overwrites the bytes of ONE logical context page after the
# append and reports which published streams move, page by page over a context that spans two
# pages, (b) prints the page table the round publishes, the page indices the append writes and the
# page indices the attention reads, and (c) re-seeds the target-tap field alone as the positive
# control on the walk's input path. It is a WIRING measurement on the drafter alone - accepted/round
# against a real target is `test_dspine_ab_real.cpp`, not this - and it loads the executor's own
# shard, so it owns the card. Device bring-up and the load path are
# test_dspine_exec_fidelity.cpp's verbatim (D-47). Opt-in by NINFER_TEST_ARTIFACT +
# NINFER_TEST_DEVICE (D-29), skips exit 77.
ninfer_add_test(ninfer_qwen3_5_dspine_context_feed_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dspine_context_feed.cpp"
          "${PROJECT_SOURCE_DIR}/src/models/common/dspine/dspine_serial_tail.cpp"
  LIBRARIES ninfer_model_loading ninfer_ops ninfer_model_execution)
set_tests_properties(ninfer_qwen3_5_dspine_context_feed_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "real;dspine;gpu" TIMEOUT 1800)
