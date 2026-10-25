"""Display-only group keys resolved from saved dataset configuration.

Never decode a numerical group ID as bits: IDs can be assigned only to observed
combinations. Decode the saved group string in the configured attribute order.
"""
import json
from pathlib import Path
import textwrap


def _read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def group_labels_for_run(run, group_names, *, _seen=None):
    """Map original group strings to meanings; unknown metadata stays unchanged."""
    run = Path(run).resolve()
    seen = set() if _seen is None else _seen
    if run in seen:
        raise ValueError('Cyclic source_run references in group label metadata')
    seen.add(run)
    context = _read(run / 'audit/audit_context.json')
    source = context.get('source_run')
    if source:
        source = Path(source)
        return group_labels_for_run(source if source.is_absolute() else run/source,
                                    group_names, _seen=seen)
    config = _read(run / 'config.json')
    info = _read(run / 'data/data_info.json')
    dataset = config.get('dataset', {})
    name = config.get('dataset_name', info.get('dataset_name', dataset.get('name')))
    options = config.get('dataset_kwargs', dataset.get('kwargs', {}))
    columns, values = [], {}
    if name == 'adult':
        columns = options.get('protected_cols', ['gender', 'race', 'native-country'])
        lower, upper = options.get('age_lower', 25), options.get('age_upper', 60)
        values = {'gender': {'0': 'Male', '1': 'Female'},
                  'native-country': {'0': 'United States', '1': 'non-US'},
                  'age_binary': {'0': f'age {lower}–{upper} (inclusive)',
                                 '1': f'age <{lower} or >{upper}'}}
        if options.get('binary_attrs', True):
            values['race'] = {'0': 'White', '1': 'non-White'}
    elif name == 'compas':
        columns = ['race', 'sex']
        values = {'race': dict(enumerate(['Caucasian', 'African-American', 'Hispanic',
                                         'Asian', 'Native American', 'Other'])),
                  'sex': {'0': 'Male', '1': 'Female'}}
        values['race'] = {str(k): v for k, v in values['race'].items()}
    elif name == 'synthetic_data':
        meta = _read(run / 'synthetic_data/metadata.json')
        meanings = meta.get('group_meaning', {})
        columns = ['S1', 'S2']
        values = {col: {str(i): f'{meanings.get(col, col)}={i}' for i in (0, 1)}
                  for col in columns}
    elif name == 'csv_tabular':
        columns = options.get('protected_cols', [])
    result = {}
    for code in map(str, group_names.values()):
        parts = code.split('_')
        if len(parts) == 1 and len(code) == len(columns) and set(code) <= {'0', '1'}:
            parts = list(code)
        if columns and len(parts) == len(columns):
            result[code] = ' / '.join(values.get(col, {}).get(value, f'{col}={value}')
                                      for col, value in zip(columns, parts))
    return result


def labels_for_frame(frame):
    """Resolve run-backed plotting data; reject conflicting meanings across runs."""
    result = {}
    if 'run' not in frame:
        return result
    for run, part in frame.groupby('run', sort=False):
        names = dict(zip(part.group_id.astype(str), part.group.astype(str)))
        for code, label in group_labels_for_run(run, names).items():
            if code in result and result[code] != label:
                raise ValueError(f'Group {code} changes meaning across runs.')
            result[code] = label
    return result


def add_group_key(fig, labels, groups=None):
    """Append a wrapped, layout-aware key while preserving titles and legends."""
    labels = labels or {}
    groups = list(dict.fromkeys(map(str, groups))) if groups is not None else list(labels)
    entries = [f'{code} = {labels[code]}' for code in groups
               if code in labels and labels[code] != code]
    if not entries:
        return
    heading = fig._suptitle
    title = heading.get_text() if heading is not None else ''
    size = heading.get_fontsize() if heading is not None else 9
    width = max(20, int(fig.get_figwidth() * 11 * 9 / size))
    key = textwrap.fill('Groups: ' + ';  '.join(entries), width=width)
    title = '\n'.join(textwrap.fill(line, width=width) for line in title.splitlines())
    fig.suptitle((title + '\n' if title else '') + key, fontsize=size)
    fig.group_key = key
