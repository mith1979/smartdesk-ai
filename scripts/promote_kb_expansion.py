"""Review and promote the 14 synthetic KB expansion drafts.

Input: data/raw/synthetic/kb_expansion/**/*.md and current data/kb/**/*.md.
Output: a preview; --approve copies drafts into data/kb as reviewed documents.
FDE connection: Week 2 document preparation and metadata quality gates;
    Week 3 requires evaluation after the KB changes.
Side effects: --approve creates new Markdown files. No existing KB file is
    overwritten, and no index, model API, or ticket API is used.
Review requirement: use --approve only after reading and accepting the drafts.
"""
import argparse
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGING = PROJECT_DIR / 'data/raw/synthetic/kb_expansion'
KB = PROJECT_DIR / 'data/kb'


def metadata(text, path):
    """Read the simple, flat frontmatter format used by this KB."""
    match = re.match(r'\A---\n(.*?)\n---\n', text, re.S)
    if not match:
        raise ValueError(f'Missing metadata: {path}')
    fields = {}
    for line in match[1].splitlines():
        key, separator, value = line.partition(':')
        if not separator or key.strip() in fields:
            raise ValueError(f'Invalid or duplicate metadata field: {path}')
        fields[key.strip()] = value.strip().strip('"')
    return fields


def main():
    parser = argparse.ArgumentParser(description='Preview or promote KB drafts')
    parser.add_argument('--approve', action='store_true',
                        help='Confirm you reviewed these drafts and add them to the KB')
    args = parser.parse_args()
    drafts = sorted(STAGING.rglob('*.md'))
    if len(drafts) != 14:
        raise ValueError(f'Expected 14 staged drafts; found {len(drafts)}')

    # Validate the entire batch before creating any target document.
    seen = set()
    for path in sorted(KB.rglob('*.md')):
        doc_id = metadata(path.read_text(encoding='utf-8'), path)['id']
        if doc_id in seen:
            raise ValueError(f'Duplicate existing document ID: {doc_id}')
        seen.add(doc_id)
    planned = []
    for path in drafts:
        text = path.read_text(encoding='utf-8')
        fields = metadata(text, path)
        target = KB / path.relative_to(STAGING)
        if target.exists():
            raise FileExistsError(f'Will not overwrite existing file: {target}')
        if fields.get('source_type') != 'synthetic' or fields.get('review_status') != 'draft':
            raise ValueError(f'Expected a synthetic draft: {path}')
        doc_id = fields['id']
        if doc_id in seen:
            raise ValueError(f'Duplicate document ID: {doc_id}')
        seen.add(doc_id)
        parts = re.split(r'^##\s+(.+)$', text, flags=re.M)
        if len(parts) < 3 or any(not parts[i+1].strip() for i in range(1,len(parts),2)):
            raise ValueError(f'Missing or empty sections: {path}')
        if len(re.findall(r'^review_status: draft$', text, re.M)) != 1:
            raise ValueError(f'Ambiguous review status: {path}')
        planned.append((target, re.sub(r'^review_status: draft$',
                                      'review_status: reviewed', text, count=1, flags=re.M)))
        print(f'{doc_id} -> {target.relative_to(PROJECT_DIR)}')

    if not args.approve:
        print('\nPreview only. Read the drafts, then run with --approve to promote them.')
        return
    # Exclusive creation prevents overwriting a file introduced after validation.
    # If an I/O error interrupts this batch, inspect the reported files before rerunning.
    for target, text in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as output:
            output.write(text)
        print(f'Added: {target.relative_to(PROJECT_DIR)}')
    print('\nAdded 14 reviewed documents. Rebuild chunks and the index next.')


if __name__ == '__main__':
    main()
