import nvtabular as nvt
from merlin.dag import ColumnSelector

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


def workflow_v2():
    price_feat = ColumnSelector(['price']) >> nvt.ops.LambdaOp(lambda x: x + 1) >> \
                 nvt.ops.Clip(min_value=1, max_value=10000) >> nvt.ops.LogOp() >> \
                 nvt.ops.LambdaOp(lambda x: x.astype("int32"))
    cat_feats = ColumnSelector(['item', 'brand']) + price_feat >> nvt.ops.Categorify()

    features = ColumnSelector(['session_id', 'time']) + cat_feats
    groupby_features = features >> nvt.ops.Groupby(
        groupby_cols=["session_id"],
        sort_cols=["time"],
        aggs={
            'item': ["list"],
            'brand': ['list'],
            'price': ['list'],
        },
        name_sep="-")
    item_feat = groupby_features['item-list'] >> nvt.ops.TagAsItemID()

    groupby_features_list = item_feat + groupby_features['brand-list', 'price-list']
    groupby_features_truncated = groupby_features_list >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH)

    return nvt.Workflow(groupby_features_truncated)
