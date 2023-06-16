import numpy as np
import nvtabular as nvt
from merlin.dag import ColumnSelector
from merlin.schema import Tags

SESSIONS_MAX_LENGTH = 20


def train_workflow_v1():
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


def eval_workflow_v1():
    item_list = ColumnSelector(['item-list'])
    features = item_list >> nvt.ops.ListSlice(0, end=-1)
    last_item = item_list >> nvt.ops.ListSlice(-1) >> nvt.ops.Rename(name='last-item')
    return nvt.Workflow(features + last_item)