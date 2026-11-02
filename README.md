# Optional local decision models

**Status: experimental.** Interfaces may change. See [SUPPORT.md](SUPPORT.md) for the support policy.

This candidate adds H2O Lightning 4B and d3-mini to Amplifier's existing optional
decision capability. Both expose bounded predicate, choice and ordinal score
answers. They do not enter the ordinary chat-model list or its fallback chain.
An ordinary provider remains responsible for planning, text writing and any
configured selection fallback. Installing this package changes no defaults.

The default installation is a small authenticated HTTP client. PyTorch, model
weights and an inference service are optional and independently operated. The
real hook module below can be composed by the CLI or another Amplifier host;
the controller has no dependency on this package or a specific model family.

```yaml
hooks:
  - module: hooks-local-decisions
    source: git+https://github.com/microsoft/amplifier-module-provider-local-decisions@main
    config:
      api_key: ${LOCAL_DECISIONS_API_KEY}
      base_url: https://your-inference-host:18990/v1/systemone
      # Optional private service CA; never disable TLS verification.
      tls_ca_file: /private/local-decisions-ca.crt
      decisions:
        enabled: true
        backends:
          local-selection:
            enabled: true
            model: lightning-int8
tools:
  - module: tool-ui-use
    config:
      selection:
        browser:
          mode: decision
          backend: local-selection
          allow_external_state: true
          fallback: standard
```

Use a host that expands credential references in hook configuration, or omit
`api_key` and set `LOCAL_DECISIONS_API_KEY` in the process environment.
Keep the UI-use behavior and browser/computer permissions configured separately.
`allow_external_state` authorizes transfer to the configured inference process,
including one on your own machine. It does not authorize UI actions. Explicit
`mode: standard` always chooses the ordinary model. Missing consent, an unavailable
service, mismatched checkpoint or an invalid answer cannot authorize an action.

The same account/token supports multiple named backends. Available model IDs are
`lightning-int8`, `d3-mini-int8`, `lightning-bf16`, and `d3-mini-bf16`. Discovery is
offline; it describes supported variants, not proof that the operator has loaded
them. An unloaded variant is unavailable and follows the consumer's configured
fallback. d3-mini remains experimental for UI tasks despite strong static results.

## Running the optional service

The lightweight provider depends on the shared library's canonical Git `main`
branch; no shared-library PyPI release is required. Install it without inference
dependencies in the consuming host:

```sh
uv pip install 'amplifier-module-provider-local-decisions @ git+https://github.com/microsoft/amplifier-module-provider-local-decisions@main'
```

For the optional service, install this package with `server` and `inference`
extras in a separate environment. Select an appropriate PyTorch
CUDA wheel for your hardware. The reference qualification used Torch 2.13/cu130,
Transformers 5.17, bitsandbytes 0.50.2 and an ARM64 NVIDIA GB10. No optimized vLLM
performance, Apple support or consumer GPU performance is claimed by this trial.
Use the platform's normal process/service manager; the provider never starts a
server, downloads a model, or upgrades dependencies on a user's behalf.

```sh
amplifier-local-decisions download --model lightning-int8 --model d3-mini-int8 --directory /private/models
amplifier-local-decisions serve --model lightning-int8 --model d3-mini-int8 --directory /private/models --token-file /private/service-token
```

Create a private token file with at least 32 random characters and mode 0600.
The default listener is literal loopback HTTP. Non-loopback listeners require
`--tls-cert` and `--tls-key`; use an authenticated private service, not an exposed
public endpoint. The lightweight client permits HTTP only for literal loopback
addresses, rejects redirects, and uses the saved account's token and CA settings.

Downloads use immutable publisher revisions and verify both LFS SHA256 and Git
blob hashes. Startup verifies the files again before importing publisher code.
Lightning's two-way readout also runs six publisher parity probes. INT8 keeps
the trained decision readout in FP32. Current supported text inputs and OCR
choices do not include image URLs, arbitrary code, boxes or UI execution.

The service bounds body size, questions, context and concurrency. It makes no
inference retries. A cancelled/timed-out client leaves a single in-flight GPU
operation holding the busy slot until it finishes; additional callers receive
an unavailable result rather than creating an unbounded queue. Ordinary fallback
is a consumer policy. CPU BF16 is available for experiments but was too slow for
interactive UI selection in the initial eight-core reference tests.

## Cost and qualification

Local decision inference has **$0 API charge**. Receipts explicitly identify
`cost.api_cost_usd: 0`, `basis: self_hosted`, and `infrastructure_included: false`.
The token count measures work; it is not multiplied by a hosted model's price.
Hardware, electricity and administration are not priced here. Hosted planning,
writing and fallback still have their own attributed charges. A local selector
does not by itself make the entire agent offline or free.

The initial frozen development benchmark favored Lightning INT8 with ordinary
fallback; no local selector qualified as a standalone UI agent. d3-mini's good
static decisions did not translate into reliable long tasks. The UI controller
therefore retains ordinary fallback and bounded recovery. Small synthetic trials
are not broad site reliability or calibrated confidence guarantees.

Pinned model sources:
[H2O Lightning](https://huggingface.co/h2oai/h2o-lightning-4b/tree/d922608ae0abcd8de27948f004ecbb8f0f2334f9),
[d3-mini](https://huggingface.co/vllm-sr/d3-mini/tree/bd0090dd758d32ba59ef5c749910a8e87fc4391f).
Publisher weights and downloaded code retain their own licenses.

## Hook lifecycle and saved accounts

The direct `hooks:` composition above uses Core's existing hook contract. The
wheel contains both `provider-local-decisions` for account metadata and the separate
`hooks-local-decisions` runtime package, declared with module type `hook`.
It requires the shared library revision providing
`amplifier_decisions.hooks.mount_account`; qualify these dependency revisions
together. Installing a key or package alone never selects a decision model.

| Composition on Core 2.0.2 | Result |
| --- | --- |
| Genuine hook, decisions enabled | Registers only named decision backends; teardown removes and closes them. |
| Genuine hook, decisions disabled | No account transport or backend is created. |
| Direct `provider-local-decisions` row, enabled or disabled | Unsupported: Core requires a chat provider mount. Do not depend on residual capability state after rejection. |
| Saved logical provider account in a host with the decision-account adapter | Host resolves that account's existing auth and exact source, then projects the genuine hook for this runtime. |

Unified's saved-account projection is a host integration, not a Core feature or
an automatic behavior of every CLI/host. Other hosts can use the direct hook
example. Ordinary chat providers remain strict, and omitting these hooks leaves
ordinary execution unchanged. The earlier proposed capability-only Core change
([Core #124](https://github.com/microsoft/amplifier-core/pull/124)) is not required
by this hook implementation and is not claimed to be implemented.

Account mounting unwinds every owned registration and transport on a partial
failure. Another account's backend or replacement is preserved. A projecting
host must verify the private lifecycle receipt before execution; only explicit
host acceptance after complete rollback permits ordinary-model fallback.
Cancellation and incomplete rollback abort initialization. The hook preserves
cancellation; Core 2.0.2's Rust session boundary wraps it as an initialization
error. No decision or UI action is replayed.

Offline qualification covers built-wheel entrypoints, the real Core 2.0.2 hook
validator/session, disabled and multiple accounts, partial failure, callback
failure, cancellation and teardown. A separate source-selection check uses the
real Foundation resolver and host account adapter. These checks use synthetic
keys and no model, API inference or device calls; they do not claim new live
inference, hardware or general website reliability results.

## Contributing

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
