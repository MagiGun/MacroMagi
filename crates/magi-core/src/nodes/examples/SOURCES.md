# Sources of the example datasets

Most example datasets are written for MacroMagi by `make_examples.py`. These are
taken, shortened and converted from open datasets by `open_data.py`:

| File | Source | Licence | What was taken |
| --- | --- | --- | --- |
| `basics.md` | [Cosmopedia 100k](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k) by Hugging Face | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Texts for children, school and general readers, without headings or tables, cut to 4,000 characters. |
| `conversations.jsonl` | [OpenAssistant oasst_top1 (2023-08-25)](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25) by OpenAssistant | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Short English conversations. |
| `math.jsonl` | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) by OpenAI (Cobbe et al., 2021) | [MIT](https://opensource.org/license/mit) | Training problems: the steps as reasoning at effort *superior*, the last step at *standard*, the answer as the response. |
| `python.jsonl` | [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) by Google Research (Austin et al., 2021) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | The training, validation and prompt problems with one test each, and their solutions. |
| `python.jsonl` | [self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) by BigCode | [ODC-By](https://opendatacommons.org/licenses/by/1-0/) | Short instructions and answers. |
| `functions.jsonl` | [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) by Glaive AI | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Short conversations with tool calls, in MacroMagi's format (`tools`, `tool_calls`, `tool` messages). |

The samples are changed only in form: shortened, filtered for length and language,
and written as MacroMagi's JSON lines. The licences above apply to them.
