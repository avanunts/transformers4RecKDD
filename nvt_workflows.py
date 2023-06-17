import numpy as np
import nvtabular as nvt
from merlin.dag import ColumnSelector
from merlin.schema import Tags

SESSIONS_MAX_LENGTH = 20


def workflow_v1():
    cat_feats = ColumnSelector(['item']) >> nvt.ops.Categorify()
    features = ColumnSelector(['session_id', 'time']) + cat_feats

    groupby_features = features >> nvt.ops.Groupby(
        groupby_cols=["session_id"],
        sort_cols=["time"],
        aggs={
            'item': ["list"],
        },
        name_sep="-")

    item_feat = groupby_features['item-list'] >> nvt.ops.TagAsItemID()
    groupby_features_list = item_feat
    groupby_features_truncated = groupby_features_list >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH)

    return nvt.Workflow(groupby_features_truncated)
