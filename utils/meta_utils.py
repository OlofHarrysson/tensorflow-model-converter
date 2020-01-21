from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.absolute()


def get_model_outdir(model_path, tf_version):
    mp = Path(model_path).stem
    tf_v = tf_version.replace('.', '_')
    # out_dir = Path('output_models') / f'{mp}_tensorflow_{tf_v}'
    out_dir = Path('output_models') / '{}_tensorflow_{}'.format(mp, tf_v)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def remove_keys(obj, key2remove):
    ''' Recursively deletes key from input dictionary '''

    if isinstance(obj, dict):
        new = obj.__class__()
        for k, v in obj.items():
            if k == key2remove:
                continue
            new[k] = remove_keys(v, key2remove)

    elif isinstance(obj, (list, set, tuple)):
        new = obj.__class__(remove_keys(v, key2remove) for v in obj)
    else:
        return obj
    return new
