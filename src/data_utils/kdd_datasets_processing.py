import numpy as np
import pandas as pd

NAN_BRAND = 'nanbrandnan'
N_MIN_BRAND_FR = 5

'''
train_sessions and test_sessions formats: {'prev_items': np.array of dtype str, 'next_item': str}
'''


def process_kdd_datasets(kdd_train_sessions, kdd_test_sessions, product_attributes):
    kdd_train_items = kdd_train_sessions.apply(concat_next_item, axis=1)
    kdd_test_items_trunc = kdd_test_sessions.prev_items  # do not use last item, because we will evaluate on it
    kdd_test_items_full = kdd_test_sessions.apply(concat_next_item, axis=1)

    train_items = pd.concat([kdd_train_items, kdd_test_items_trunc])
    eval_items = kdd_test_items_full

    product_attributes.fillna({'brand': NAN_BRAND}, inplace=True)
    product_features = get_products_features(product_attributes)

    train_ds = get_ds_from_items_and_product_features(train_items, product_features)
    eval_ds = get_ds_from_items_and_product_features(eval_items, product_features)
    return train_ds, eval_ds


def get_products_features(product_attributes):
    brand_low = product_attributes['brand'].apply(lambda x: x.lower())
    brand_low_and_frequent = get_frequent_brands(brand_low.values)
    return pd.DataFrame({
        'item': product_attributes['id'],
        'brand': brand_low_and_frequent,
        'price': product_attributes['price']
    })


def get_frequent_brands(brands):
    brand_val, brand_count = np.unique(brands, return_counts=True)
    fr_brands = set(brand_val[brand_count >= N_MIN_BRAND_FR])
    process_brand = np.vectorize(lambda x: x if x in fr_brands else NAN_BRAND)
    return process_brand(brands)


def get_ds_from_items_and_product_features(items, product_features):
    time_and_item_seq = items.apply(append_time)
    session_ids = np.arange(len(items))

    ds = pd.DataFrame({'session_id': session_ids, 'time_and_item_seq': time_and_item_seq}).explode('time_and_item_seq')
    ds['time'] = ds['time_and_item_seq'].apply(lambda x: x[0])
    ds['item'] = ds['time_and_item_seq'].apply(lambda x: x[1])
    ds.drop(columns=['time_and_item_seq'], inplace=True)

    ds = pd.merge(ds, product_features, on='item', how='left')
    return ds


def concat_next_item(row):
    return np.append(row['prev_items'], row['next_item'])


def append_time(items_row):
    return list(enumerate(items_row))
