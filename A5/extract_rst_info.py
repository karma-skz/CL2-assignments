import xml.etree.ElementTree as ET
import glob
import json

def extract_rst_info(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    relation_types = {}
    header = root.find('header/relations')
    if header:
        for rel in header.findall('rel'):
            relation_types[rel.get('name')] = rel.get('type')
    
    segments = []
    body = root.find('body')
    if body:
        for seg in body.findall('segment'):
            seg_id = seg.get('id')
            parent_id = seg.get('parent')
            relname = seg.get('relname', '')
            
            if relname == 'span': # "span" means this segment is the nucleus of its parent
                is_nucleus = True
                is_satellite = False
                rel_type = 'span'
            elif relname: # has a named relation to parent
                rel_type = relation_types.get(relname, 'unknown')
                # for RST relations the child is a satellite
                # for multinuc relations its a nucleus
                is_satellite = (rel_type == 'rst')
                is_nucleus = (rel_type == 'multinuc')
            else: # probably root if no rel name
                is_nucleus = True
                is_satellite = False
                rel_type = 'root'
            
            segments.append({
                'id': seg_id,
                'text': seg.text.strip() if seg.text else '',
                'parent': parent_id,
                'relation': relname if relname else 'root',
                'relation_type': rel_type,
                'is_nucleus': is_nucleus,
                'is_satellite': is_satellite
            })

    return {
        'file': filepath.split('/')[-1],
        'relation_types': relation_types,
        'segments': segments
    }

def main():
    results = {}
    for rs3_file in sorted(glob.glob('[0-9]*.rs3'), key=lambda x: int(x.replace('.rs3', ''))):
        para_num = rs3_file.replace('.rs3', '')
        results[para_num] = extract_rst_info(rs3_file)
    
    with open('rst_extracted_info.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    for para_num, data in sorted(results.items(), key=lambda x: int(x[0])):
        nucleus_count = sum(1 for s in data['segments'] if s['is_nucleus'])
        satellite_count = sum(1 for s in data['segments'] if s['is_satellite'])
        
        print(f"\nParagraph {para_num}:")
        print(f"  Segments: {len(data['segments'])} "
              f"(Nucleus: {nucleus_count}, Satellite: {satellite_count})")
        print(f"  Relations: {', '.join(sorted(data['relation_types'].keys()))}")
        
        for seg in data['segments']:
            role = "N" if seg['is_nucleus'] else "S"
            print(f"    [{role}] Seg {seg['id']}: {seg['relation']} → parent {seg['parent']}")

if __name__ == '__main__':
    main()