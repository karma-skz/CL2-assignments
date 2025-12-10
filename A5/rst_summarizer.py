#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import glob
import re
import os

class RSTSummarizer:
    RELATION_WEIGHTS = {
        'Attribution': 2, 'Background': 5, 'Condition': 4, 'Contrast': 7, 'Elaboration': 3, 'Explanation': 6, 'Joint': 7, 'Temporal': 6, 'span': 5, 'same-unit': 6
    }
    
    def __init__(self):
        self.segments_by_id = {}
        self.groups_by_id = {}
        self.parent_map = {}
        self.relation_types = {}
    
    def parse_rs3_file(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        self.segments_by_id = {}
        self.groups_by_id = {}
        self.parent_map = {}
        
        header = root.find('header') or root
        relations = header.find('relations') or header
        for rel in relations.findall('rel'):
            self.relation_types[rel.get('name')] = rel.get('type')
        
        body = root.find('body') or root
        for segment in body.findall('segment'):
            seg_id = segment.get('id')
            self.segments_by_id[seg_id] = {
                'id': seg_id,
                'parent': segment.get('parent'),
                'relname': segment.get('relname'),
                'text': segment.text.strip() if segment.text else '',
                'type': 'segment'
            }
            self.parent_map[seg_id] = segment.get('parent')
        
        for group in body.findall('group'):
            group_id = group.get('id')
            self.groups_by_id[group_id] = {
                'id': group_id,
                'type': group.get('type'),
                'parent': group.get('parent'),
                'relname': group.get('relname')
            }
            if group.get('parent'):
                self.parent_map[group_id] = group.get('parent')
    
    def get_depth(self, node_id):
        depth = 0
        current = node_id
        visited = set()
        
        while current in self.parent_map and current not in visited:
            visited.add(current)
            current = self.parent_map[current]
            depth += 1
        
        return depth
    
    def is_nucleus(self, node_id):
        if node_id in self.segments_by_id:
            relname = self.segments_by_id[node_id].get('relname')
        elif node_id in self.groups_by_id:
            relname = self.groups_by_id[node_id].get('relname')
        else:
            return True
        
        if not relname or relname == 'span':
            return True
        
        rel_type = self.relation_types.get(relname, 'unknown')
        
        if rel_type == 'multinuc':
            return True
        if rel_type == 'rst':
            return False
        return True
    
    def calculate_segment_score(self, seg_id):
        segment = self.segments_by_id[seg_id]
        score = 0
        
        if self.is_nucleus(seg_id):
            score += 5
        
        depth = self.get_depth(seg_id)
        score += max(0, 10 - depth)
        
        relname = segment.get('relname', '')
        if relname:
            score += self.RELATION_WEIGHTS.get(relname, 0)

        text_len = len(segment['text'])
        if text_len > 50:
            score += 2
        if text_len > 100:
            score += 2
        
        return score
    
    def select_summary_segments(self, max_segments=2):
        segment_scores = []
        for seg_id, segment in self.segments_by_id.items():
            score = self.calculate_segment_score(seg_id)
            text = segment['text']
            
            if text and text[-1] in '.!?':
                score += 3
            
            word_count = len(text.split())
            if word_count >= 8:
                score += 2
            
            segment_scores.append((score, seg_id, segment))
        
        segment_scores.sort(reverse=True)
        
        selected = []
        for score, seg_id, segment in segment_scores[:max_segments]:
            selected.append({
                'id': seg_id,
                'text': segment['text'],
                'score': score,
                'is_nucleus': self.is_nucleus(seg_id),
                'depth': self.get_depth(seg_id),
                'relation': segment.get('relname', 'N/A')
            })
        
        return selected
    
    def generate_summary(self, selected_segments):
        selected_segments.sort(key=lambda x: int(x['id']))
        
        texts = []
        for seg in selected_segments:
            text = seg['text']
            if text and text[-1] not in '.!?,;:':
                text += '.'
            texts.append(text)
        
        summary_text = ' '.join(texts)
        summary_text = summary_text.replace('..', '.')
        summary_text = ' '.join(summary_text.split())
        
        return summary_text
    
    def summarize_file(self, filepath, max_segments=2):
        self.parse_rs3_file(filepath)
        selected = self.select_summary_segments(max_segments)
        summary = self.generate_summary(selected)
        
        return {
            'file': os.path.basename(filepath),
            'selected_segments': selected,
            'summary': summary
        }

def load_paragraphs(paragraph_file='paragraph.txt'):
    paragraphs = {}
    with open(paragraph_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                parts = line.split('. ', 1)
                paragraphs[parts[0]] = parts[1]
    return paragraphs


def main():
    summarizer = RSTSummarizer()
    paragraphs = load_paragraphs()
    rs3_files = sorted(glob.glob('[0-9]*.rs3'), key=lambda x: int(x.replace('.rs3', '')))
    
    llm_summaries = {
        '1': "The author argues that California Common Cause should focus on its core mission of governmental structure and process rather than spreading resources thin by supporting every popular cause like the Nuclear Freeze Initiative.",
        '2': "Syncom diskettes use four protective mechanisms—cleaning agents, lubricants, carbon additives, and strong binders—along with a special jacket liner to prevent dust and particles from causing data errors.",
        '3': "Creating a Victorian-style flower arrangement requires combining plants of varying heights, shapes, and colors, with attention to diverse leaf textures ranging from silver dusty miller to colorful coleus.",
        '4': "Long lines of job seekers demonstrate a shortage of available employment rather than a lack of effort, contradicting claims that unemployment stems from insufficient initiative.",
        '5': "When real estate agents asked the author's English wife what she would call an elaborate chandelier, she wittily replied \"ostentatious\" after discussing other British-American vocabulary differences.",
        '6': "Mother Teresa advised visiting American teachers to smile at their spouses, and when asked if marriage made her advice relevant, revealed she is \"married\" to Jesus, who she finds can be very demanding.",
        '7': "ZPG's Urban Stress Test ranks 184 U.S. cities using eleven population-related criteria to help citizens and officials understand urban pressures in an accessible format.",
        '8': "Archaeological study of ancient shipwrecks provides uniquely valuable chronological data since all objects represent a single moment in time, and underwater conditions preserve perishable items that would decay on land.",
        '9': "A student pilot lost control of her spinning Cessna 150 but survived by remembering her instructor's counterintuitive advice to simply release the controls and let the plane stabilize itself.",
        '10': "This paper compares Rhetorical Structure Theory, which analyzes texts through functional relations between parts, with the broader Systemic Linguistics approach, which categorizes texts by the processes they perform."
    }
    
    results = []
    for rs3_file in rs3_files:
        para_num = rs3_file.replace('.rs3', '')
        result = summarizer.summarize_file(rs3_file, max_segments=3)
        result['para_num'] = para_num
        result['original'] = paragraphs.get(para_num, '')
        results.append(result)
    
    with open('rst_summarization_report.txt', 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"PARAGRAPH {result['para_num']}\n")
            f.write(f"Original: {result['original']}\n")
            f.write(f"Summary: {result['summary']}\n")
            f.write(f"LLM: {llm_summaries.get(result['para_num'], '')}\n\n")

if __name__ == '__main__':
    main()
