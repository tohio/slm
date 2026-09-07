# Read-only web search in chat

Guide for preparing tool-use SFT data and running one bounded search interaction
through the existing SLM chat CLI. This is plumbing support, not a claim that an
untuned model or Mini reliably knows when or how to use tools.

## Conversation contract

The model-facing tool is always `web_search(query)`, independent of the search
provider. Assistant requests use existing vocabulary delimiters:

```text
<|tool|>{"name":"web_search","arguments":{"query":"a useful search query"}}<|endoftool|>
```

The assistant's content must be exactly one complete call. The dispatcher
validates the name, exact argument keys, and a nonempty query of at most 256
characters/75 words. It does not evaluate code or import model-named functions.
A result is a `role: tool` message whose JSON content contains `name`, `status`
(`ok`, `empty`, or `error`), normalized `results`, and `calls_remaining: 0`.
The next assistant message must be a final answer, not another call.

`config/chat.py` owns the shared templates. Tool results are outside the
assistant generation/loss mask; assistant requests and final answers are inside
it. External result text is JSON-escaped so it cannot inject vocabulary control
tokens. The prompt treats results as untrusted evidence, never instructions;
this does not guarantee that a model will resist all natural-language injection.
Ordinary no-tool rendering is unchanged when no tools are supplied.

The tokenizer's existing tool tokens are checked before enabling the template.
Only the rendering template changes: no vocabulary growth, tokenizer retraining,
or embedding reinitialization. The active template is saved with an SFT model.

## SFT input

Continue to use conversational JSONL (`messages` or `conversations`) and the
existing SFT preparation/trainer. Supply complete user → assistant call → tool
result → assistant final interactions, along with direct-answer examples.
Provider credentials and API-specific details do not belong in training data.

An optional reviewed local file can be merged into instruct data before the
existing grouped train/validation split:

```bash
make prepare-sft SIZE=mini SFT_TOOL_DATA=/path/to/reviewed-tool-data.jsonl
```

The local file's SHA-256 is recorded in the SFT source contract. Normal source
validation, deduplication, and grouped prompt isolation still apply. When this
option is used, no-tool examples also expose the tool schema, teaching that a
direct answer can be appropriate. Changing the local file requires the same
intentional `--force` preparation used for other source changes. Training-size
caps still apply; inspect the selected examples and preflight audit.

`finetune/examples/tool_conversations.jsonl` contains **six illustrative format
examples**, including success, query wording, empty/error responses, and no-tool
answers. Its example.org sources and fictional entities are not live evidence or
a production training corpus. It is never included by default. Build substantive
reviewed data outside this repository rather than treating the examples as a
capability-training dataset.

The existing SFT preflight can check tokenization/masks without optimizing:

```bash
.venv/bin/python finetune/train_sft.py \
  --config finetune/configs/sft_instruct_mini.yaml --preflight-only
```

## Runtime

Choose an operator-configured provider in `.env` before enabling search:

```text
WEB_SEARCH_PROVIDER=brave
BRAVE_SEARCH_API_KEY=<provider key>
```

Alternatively, use `WEB_SEARCH_PROVIDER=searxng` with a JSON-enabled
`SEARXNG_URL=https://your-instance.example/search`. HTTPS is required except for
loopback development endpoints. The model cannot choose the endpoint. The two
small adapters normalize their responses to the same model-facing schema;
additional providers can implement that same callable without changing model data.

```bash
.venv/bin/python inference/chat.py \
  --model "$RESULTS_DIR/runs/mini/sft_instruct/final" \
  --web-search --max-new-tokens 256
```

Without `--web-search`, no search provider is configured or contacted. With it,
one generated query per user turn may be sent to the selected external service.
Do not enable search for requests whose contents must remain local.

The HTTP adapters have connect/read timeouts, an overall read deadline, a
256,000-byte response limit, no redirects/retries/pagination, and at most three
normalized results (2,400 characters total before the small result envelope).
Only snippets are used; result URLs are not fetched. Empty/failed searches are
returned explicitly without provider exception text or credentials.

The runtime never executes a second tool request in the same turn. For an
oversized conversation it drops complete older turns, preserving system/tool
instructions and the current interaction; it rejects a current interaction that
still exceeds the model context instead of truncating away those boundaries.
History is committed only after the turn completes successfully.

Mini can check parsing, dispatch, result reinsertion, and continuation. Evaluate
actual search selection, factual grounding, refusals, and answer quality with the
larger models. Merely installing the runtime does not confer learned tool use.

## See Also

- [Inference](../inference/README.md), [SFT](../finetune/README.md)
- [TRL SFT tool support](https://huggingface.co/docs/trl/sft_trainer#tool-calling-with-sft)
- [SearXNG search API](https://docs.searxng.org/dev/search_api.html)
- [Brave search API](https://api-dashboard.search.brave.com/documentation/services/web-search)
