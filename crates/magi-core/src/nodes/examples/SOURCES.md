# Sources of the example datasets

Most example datasets are written for MacroMagi by `make_examples.py`. These are
taken, shortened and converted from open datasets by `open_data.py`:

| File | Source | Licence | What was taken |
| --- | --- | --- | --- |
| `conversations.jsonl` | [OpenAssistant oasst_top1 (2023-08-25)](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) by OpenAssistant | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Short English conversations, leaving out those that name the assistant that wrote them. |
| `functions.jsonl` | [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) by Glaive AI | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Short conversations with tool calls, in MacroMagi's format (`tools`, `tool_calls`, `tool` messages). |

The samples are changed only in form: shortened, filtered for length and language,
and written as MacroMagi's JSON lines. The licences above apply to them.

`work.jsonl` is made by `make_work_data.py` from MacroMagi's own tools and MagiDB (the
repository's `magidb/`, CC BY 4.0 for its text, MIT for its code).
