# deepseek_v4_flash_custom_banking_reasoning_high_1trial

## Summary
- tasks: 97
- parallelism: 8
- model: openrouter/deepseek/deepseek-v4-flash
- harness_or_provider_error: 86
- provider_rate_limit: 8
- reward_failure: 1
- success: 2

## Rows
- task_001: provider_rate_limit, reward=None, termination=None, seconds=553.2, events=[kb_search=59, kb_read=56, planner_invalid_shape_retry=5, subagent_start=5, planner_internal_tool_turn=5, subagent_limit=4]
  tail: litellm.exceptions.APIError: litellm.APIError: APIError: OpenrouterException - {"error":{"message":"Insufficient credits. Add more using https://openrouter.ai/settings/credits","code":402}}
- task_002: provider_rate_limit, reward=None, termination=None, seconds=561.59, events=[kb_search=79, kb_read=70, subagent_start=6, subagent_limit=6, planner_internal_tool_turn=6, embedding_error=4]
  tail: litellm.exceptions.APIError: litellm.APIError: APIError: OpenrouterException - {"error":{"message":"Insufficient credits. Add more using https://openrouter.ai/settings/credits","code":402}}
- task_003: provider_rate_limit, reward=None, termination=None, seconds=577.57, events=[kb_read=75, kb_search=71, planner_invalid_shape_retry=9, subagent_start=7, planner_internal_tool_turn=7, subagent_limit=6]
  tail: litellm.exceptions.APIError: litellm.APIError: APIError: OpenrouterException - {"error":{"message":"Insufficient credits. Add more using https://openrouter.ai/settings/credits","code":402}}
- task_004: success, reward=1.0, termination=user_stop, seconds=381.37, events=[kb_search=6, planner_invalid_shape_retry=5, run_start=1, subagent_start=1, kb_read=1, subagent_done=1]
- task_005: reward_failure, reward=0.0, termination=user_stop, seconds=138.58, events=[planner_invalid_shape_retry=8, run_start=1, run_done=1]
- task_006: success, reward=1.0, termination=user_stop, seconds=238.39, events=[kb_search=5, kb_read=5, planner_invalid_shape_retry=2, run_start=1, subagent_start=1, subagent_done=1]
- task_007: provider_rate_limit, reward=None, termination=None, seconds=582.33, events=[kb_search=74, kb_read=70, subagent_start=8, planner_internal_tool_turn=8, subagent_limit=6, planner_invalid_shape_retry=5]
  tail: litellm.exceptions.APIError: litellm.APIError: APIError: OpenrouterException - {"error":{"message":"Insufficient credits. Add more using https://openrouter.ai/settings/credits","code":402}}
- task_008: provider_rate_limit, reward=None, termination=None, seconds=555.12, events=[kb_search=79, kb_read=39, subagent_start=7, planner_internal_tool_turn=7, subagent_limit=6, planner_invalid_shape_retry=2]
  tail: litellm.exceptions.APIError: litellm.APIError: APIError: OpenrouterException - {"error":{"message":"Insufficient credits. Add more using https://openrouter.ai/settings/credits","code":402}}
- task_010: provider_rate_limit, reward=None, termination=None, seconds=424.62, events=[kb_read=28, kb_search=17, planner_invalid_shape_retry=13, subagent_start=4, planner_internal_tool_turn=4, subagent_limit=3]
  tail: litellm.exceptions.APIError: litellm.APIError: APIError: OpenrouterException - {"error":{"message":"Insufficient credits. Add more using https://openrouter.ai/settings/credits","code":402}}
- task_012: provider_rate_limit, reward=None, termination=None, seconds=312.28, events=[kb_search=31, kb_read=12, subagent_start=3, planner_internal_tool_turn=3, subagent_limit=2, run_start=1]
  tail: litellm.exceptions.APIError: litellm.APIError: APIError: OpenrouterException - {"error":{"message":"Insufficient credits. Add more using https://openrouter.ai/settings/credits","code":402}}
- task_014: provider_rate_limit, reward=None, termination=None, seconds=201.78, events=[kb_read=17, kb_search=14, planner_invalid_shape_retry=4, subagent_start=3, subagent_done=2, planner_internal_tool_turn=2]
  tail: litellm.exceptions.APIError: litellm.APIError: APIError: OpenrouterException - {"error":{"message":"Insufficient credits. Add more using https://openrouter.ai/settings/credits","code":402}}
- task_015: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_016: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_017: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_018: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_019: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_020: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_021: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_022: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_023: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_024: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_025: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_026: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_027: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_028: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_029: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_031: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_032: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_033: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_034: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_035: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_036: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_037: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_038: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_039: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_040: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_041: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_043: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_044: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_045: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_046: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_047: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_048: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_049: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_050: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_051: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_052: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_053: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_054: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_055: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_056: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_057: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_058: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_059: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_060: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_061: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_062: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_063: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_064: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_065: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_066: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_067: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_068: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_069: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_070: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_071: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_072: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_073: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_074: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_075: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_076: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_077: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_078: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_079: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_080: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_081: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_082: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_083: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_084: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_085: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_086: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_087: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_088: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_089: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_090: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_091: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_092: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_093: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_094: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_095: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_096: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_097: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_098: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_099: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_100: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_101: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
- task_102: harness_or_provider_error, reward=None, termination=None, seconds=None, events=[]
  tail: not run: provider_credit_exhausted
