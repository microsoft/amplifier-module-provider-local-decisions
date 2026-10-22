# Optional local decision models

**Status: experimental.** Interfaces may change. See [SUPPORT.md](SUPPORT.md) for the support policy.

This candidate adds H2O Lightning 4B and d3-mini to Amplifier's existing optional
decision capability. Both expose bounded predicate, choice and ordinal score
answers. They do not enter the ordinary chat-model list or its fallback chain.
An ordinary provider remains responsible for planning, text writing and any
configured selection fallback. Installing this package changes no defaults.

The default installation is a small authenticated HTTP client. PyTorch, model
weights and an inference service are optional and independently operated. The
adapter is intended for reuse by the CLI, Unified and other UI-use hosts, subject
to the lifecycle limitation below. The controller has no dependency on this
package or a specific model family.

## Current Core compatibility limitation

Source inspection shows that this provider's capability-only mount is **not
supported by Core 2.0.2's provider lifecycle**. Core requires a newly mounted
chat-provider object. This module publishes `decision.backends` instead, so Core
rejects both an enabled mount and an explicitly listed disabled row that returns
without mounting a chat provider. For ordinary use, omit this optional provider
row; ordinary providers and UI-use selection remain unchanged.

This finding is from source inspection, not a new local-provider runtime
reproduction. A separate Jev lifecycle probe showed an executable backend
surviving Core's rejected mount until normal session cleanup. Do not assume
rejection removes a backend or treat earlier inference/benchmark results as
qualification of a supported Core mount. Consumers must not rely on rejected
mount side effects.

[Core #124](https://github.com/microsoft/amplifier-core/pull/124) proposes a
capability-only lifecycle for modules such as this one; it is a documentation
proposal, not implemented support. A supported lifecycle and failure cleanup
still need agreement and qualification. No future Core version or compatibility
floor is promised here. Installing the client or standalone inference service
does not resolve that boundary.

The following YAML illustrates the candidate configuration format; it is not a
supported production setup on Core 2.0.2:

```yaml
providers:
  - module: provider-local-decisions
    source: git+https://github.com/microsoft/amplifier-module-provider-local-decisions@main
    config:
      chat_enabled: false
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
