from typing import List, Tuple

Setting = Tuple[str, ...]


def get_strategic_plan_configs() -> List[Setting]:
    default_plan: Setting = ()

    no_hashjoin: Setting = ("SET enable_hashjoin = 'off';",)
    no_nestloop: Setting = ("SET enable_nestloop = 'off';",)
    no_mergejoin: Setting = ("SET enable_mergejoin = 'off';",)

    no_seqscan: Setting = ("SET enable_seqscan = 'off';",)
    no_indexscan: Setting = ("SET enable_indexscan = 'off';",)
    no_bitmapscan: Setting = ("SET enable_bitmapscan = 'off';",)

    return [
        default_plan,
        no_hashjoin,
        no_nestloop,
        no_mergejoin,
        no_seqscan,
        no_indexscan,
        no_bitmapscan,
    ]
