"""
data_preprocessing/scaler_utils.py

Contains functions for data standardization (using StandardScaler),
as well as saving/loading scaler objects for future use.

【去掉原先的 logit transform】, 保留/新增 bounded transform(0..100 => -1..1).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def _ensure_2d(arr):
    """
    保证传入的数组是 2D。
    - 1D -> reshape(-1, 1)
    - 2D 或更高 -> 原样返回
    """
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _normalize_output_transform_name(name):
    if name is None:
        return None
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key in {"", "none", "null"}:
        return None
    if key == "log1p":
        return "log1p"
    if key == "log":
        return "log"
    raise ValueError(f"Unsupported output transform: {name}")


def _normalize_output_transform_specs(output_transforms):
    if not output_transforms:
        return []
    if isinstance(output_transforms, dict):
        output_transforms = [output_transforms]

    normalized = []
    for spec in output_transforms:
        if not isinstance(spec, dict):
            raise TypeError("Each output transform spec must be a dict.")
        name = _normalize_output_transform_name(spec.get("name"))
        if name is None:
            continue

        cols = spec.get("cols", [])
        cols = [int(c) for c in cols]
        cols = sorted(set(cols))

        try:
            eps = float(spec.get("eps", 1e-8))
        except (TypeError, ValueError):
            eps = 1e-8
        if eps <= 0:
            eps = 1e-8

        normalized.append({"name": name, "cols": cols, "eps": eps})
    return normalized


def resolve_output_transform_specs(
    y_col_names,
    transform_name=None,
    transform_cols=None,
    transform_eps=1e-8,
):
    name = _normalize_output_transform_name(transform_name)
    if name is None:
        return []

    y_names = [str(c).strip() for c in y_col_names]
    idx_map = {name: idx for idx, name in enumerate(y_names)}

    if transform_cols is None:
        cols = list(range(len(y_names)))
    else:
        cols = []
        for item in transform_cols:
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if idx < 0 or idx >= len(y_names):
                    raise IndexError(f"Output transform index out of range: {idx}")
            else:
                key = str(item).strip()
                if key not in idx_map:
                    raise KeyError(f"Output transform column not found in targets: {key}")
                idx = idx_map[key]
            if idx not in cols:
                cols.append(idx)

    if not cols:
        return []

    try:
        eps = float(transform_eps)
    except (TypeError, ValueError):
        eps = 1e-8
    if eps <= 0:
        eps = 1e-8

    return [{"name": name, "cols": cols, "eps": eps}]


def apply_output_transforms(y, output_transforms=None):
    specs = _normalize_output_transform_specs(output_transforms)
    y_out = _ensure_2d(np.asarray(y, dtype=np.float32).copy())
    if not specs:
        return y_out

    for spec in specs:
        cols = spec["cols"]
        if not cols:
            continue
        if spec["name"] == "log1p":
            y_out[:, cols] = np.log1p(np.clip(y_out[:, cols], -1.0 + spec["eps"], None))
        elif spec["name"] == "log":
            y_out[:, cols] = np.log(np.clip(y_out[:, cols], spec["eps"], None))
        else:
            raise ValueError(f"Unsupported output transform: {spec['name']}")
    return y_out


def attach_output_transform_metadata(scaler_y, output_transforms=None):
    specs = _normalize_output_transform_specs(output_transforms)
    if not specs:
        return scaler_y

    if scaler_y is None:
        wrapped = {"type": "identity", "scaler": None}
    elif isinstance(scaler_y, dict):
        wrapped = dict(scaler_y)
    else:
        wrapped = {"type": "standard", "scaler": scaler_y}

    wrapped["output_transforms"] = specs
    return wrapped


def bounded_transform(y):
    """
    y in [0,100] => z in [-1,1], linear:
        z = 2*(y/100) - 1
    clamp y to [0,100] just in case
    """
    y_ = np.clip(y, 0, 100)
    return 2*(y_/100.0) - 1

def inverse_bounded_transform(z):
    """
    z => clamp in [-1,1], => y=100*(z+1)/2 in [0,100]
    """
    z_ = np.clip(z, -1, 1)
    return 100.0*(z_+1.0)/2.0

def standardize_data(X_train, X_val,
                     Y_train, Y_val,
                     do_input=True,
                     do_output=False,
                     numeric_cols_idx=None,
                     scale_cols_idx=None,
                     do_output_bounded=False,
                     bounded_output_cols_idx=None):
    """
    Optionally standardize input features (X) and/or output targets (Y).

    If do_output_bounded is True (and bounded_output_cols_idx is None),
    then all output columns are first transformed using bounded_transform (0..100 => -1..1)
    before applying StandardScaler.

    If bounded_output_cols_idx is provided (a list of column indices),
    then only those specified columns are processed with bounded_transform,
    and the remaining columns are processed normally.

    scale_cols_idx controls which X columns are standardized. If None,
    numeric_cols_idx is used; if that is also None, all columns are scaled.
    """
    scaler_x = None
    scaler_y = None

    X_train_s = np.copy(X_train)
    X_val_s   = np.copy(X_val)
    Y_train_s = np.copy(Y_train)
    Y_val_s   = np.copy(Y_val)

    if do_input:
        if scale_cols_idx is None:
            if numeric_cols_idx is None:
                numeric_cols_idx = list(range(X_train.shape[1]))
            scale_cols_idx = numeric_cols_idx
        scaler_x = StandardScaler()
        scaler_x.fit(X_train_s[:, scale_cols_idx])
        X_train_s[:, scale_cols_idx] = scaler_x.transform(X_train_s[:, scale_cols_idx])
        X_val_s[:, scale_cols_idx]   = scaler_x.transform(X_val_s[:, scale_cols_idx])

    if do_output:
        # 根据 bounded_output_cols_idx 优先，仅对指定列应用 bounded_transform
        if bounded_output_cols_idx is not None:
            for i in bounded_output_cols_idx:
                Y_train_s[:, i] = bounded_transform(Y_train_s[:, i])
                Y_val_s[:, i]   = bounded_transform(Y_val_s[:, i])
            transform_type = "bounded+standard"
        elif do_output_bounded:
            # 若未指定具体列，但标记为 bounded，则对所有输出列转换
            for i in range(Y_train_s.shape[1]):
                Y_train_s[:, i] = bounded_transform(Y_train_s[:, i])
                Y_val_s[:, i]   = bounded_transform(Y_val_s[:, i])
            transform_type = "bounded+standard"
        else:
            transform_type = "standard"

        # 对所有输出列进行标准化处理
        scaler_obj = StandardScaler()
        scaler_obj.fit(Y_train_s)
        Y_train_s = scaler_obj.transform(Y_train_s)
        Y_val_s   = scaler_obj.transform(Y_val_s)

        scaler_y = {
            "type": transform_type,
            "scaler": scaler_obj
        }
        # 如果指定了 bounded 输出列，则记录下来以便逆变换时使用
        if bounded_output_cols_idx is not None:
            scaler_y["bounded_cols"] = list(bounded_output_cols_idx)

    return (X_train_s, X_val_s, scaler_x), (Y_train_s, Y_val_s, scaler_y)

def save_scaler(scaler, path):
    if scaler is not None:
        joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def inverse_transform_output(y_pred, scaler_y):
    """
    把标准化/[-1,1] 域的预测反变换回真实量纲。
    1D → 自动 reshape(-1,1)；防止 StandardScaler 报错。
    """
    if scaler_y is None:
        return y_pred

    # -------- 先确保输入是 2D -------
    y_pred = _ensure_2d(y_pred.copy())

    if not isinstance(scaler_y, dict):
        return scaler_y.inverse_transform(y_pred)

    transform_type = scaler_y.get("type", "standard")
    scaler_obj = scaler_y.get("scaler", None)

    if scaler_obj is not None:
        y_ = scaler_obj.inverse_transform(y_pred)
    else:
        y_ = y_pred.copy()

    if transform_type.startswith("bounded"):
        bound_cols = scaler_y.get("bounded_cols", range(y_.shape[1]))
        for i in bound_cols:
            y_[:, i] = inverse_bounded_transform(y_[:, i])

    output_transforms = _normalize_output_transform_specs(scaler_y.get("output_transforms", None))
    for spec in reversed(output_transforms):
        cols = spec["cols"]
        if not cols:
            continue
        if spec["name"] == "log1p":
            y_[:, cols] = np.expm1(y_[:, cols])
        elif spec["name"] == "log":
            y_[:, cols] = np.exp(y_[:, cols])
        else:
            raise ValueError(f"Unsupported output transform: {spec['name']}")

    return y_
