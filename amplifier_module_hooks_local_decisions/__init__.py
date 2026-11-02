"""Optional local decisions through the supported Amplifier hook lifecycle."""
from amplifier_decisions.hooks import mount_account
from amplifier_module_provider_local_decisions import LocalDecisionsProvider

__amplifier_module_type__ = 'hook'


async def mount(coordinator, config=None):
    return await mount_account(coordinator, config, provider_factory=LocalDecisionsProvider,
                               default_id='local-decisions', hook_module='hooks-local-decisions')
