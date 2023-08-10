import json
import pickle
import re
import numpy as np
from tqdm import tqdm
from numpy import dot
from functools import cache
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, util

WORD_EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
RE_PAIRS = re.compile(r"\('(.*?)',\s*'(.*?)'\)")

def load_train():
    ent_count = {}
    words_dict = {}

    with open("data/train.txt", "r") as f:
        for line in f:
            s, p, o, *_ = line.strip().split("\t")
            ent_count[s] = ent_count.get(s, 0) + 1
            ent_count[o] = ent_count.get(o, 0) + 1

            for word in s.split("_") + o.split("_"):
                if word in ("(", ")"): continue
                sent = f'{s.replace("_", " ")} {p} {o.replace("_", " ")}'
                if word not in words_dict:
                    words_dict[word] = []
                words_dict[word].append(sent)    

    return ent_count, words_dict

def load_types():
    ent_type = {}

    with open("data/entities.txt", "r") as f:
        for line in f:
            ent, *_, typ = line.strip().split("\t")
            ent_type[ent] = typ

    return ent_type

def load_hypotheses():
    hyp = set()
    hyp_scores = {}
    hyp_labels = {}

    with open("data/hypotheses.csv", "r") as f:
        for line in f:
            s, p, o, label, scicheck_score = line.strip().split(",")
            if float(scicheck_score) > 0.5:
                h = (s, p, o)
                hyp_scores[h] = float(scicheck_score)
                hyp_labels[h] = label
                hyp.add(h)

    return list(hyp), hyp_scores, hyp_labels

'''
    Process the data, removing newlines and spaces from the names of the entities
'''
def load_entity_sentences():
    data = {}

    with open("data/entity2count.json", "r") as f:
        for line in f:
            ent_data = json.loads(line.strip())
            for ent_name, dist_data in ent_data.items():
                # only one iteration
                total = sum(dist_data.values())
                data[ent_name.strip().replace(" ", "_")] = {"years": dist_data, "total": total}
    
    return data

def load_max_cos_similarity() :
    data = {}

    with open("data/max_cos_sim.csv", "r") as f:
        for line in f:
            index, s, p, o, _, cos = line.strip().split(",")
            if not index: continue  # skip header

            s = s.replace(" ", "_") 
            p = p.replace(" ", "_") 
            o = o.replace(" ", "_") 
            data[(s, p, o)] = cos

    return data

def load_graph_embeddings():
    embeddings_raw = np.load("data/cskg_TransE_l2_entity.npy")
    embeddings = {}
    
    with open("data/entities_graph_emb.tsv", "r") as f:
        for ent_line, emb in zip(f, embeddings_raw):
            ent = ent_line.split("\t")[1].strip()
            embeddings[ent] = emb

    return embeddings

def graph_emb_similarity(s, o, embeddings):
    emb_s = embeddings.get(s)
    emb_o = embeddings.get(o)

    if emb_s is None or emb_o is None:
        return 0.0
    
    return dot(emb_s, emb_o) / (norm(emb_s) * norm(emb_o))

@cache
def encode_entity(ent):
    name = ent.replace("_", " ")
    return WORD_EMB_MODEL.encode(name)

def word_emb_similarity(s, o):
    emb_s = encode_entity(s)
    emb_o = encode_entity(o)

    return util.cos_sim(emb_s, emb_o).item()

def load_entity_pairs():
    data = {}

    with open("data/pair2freq.pkl", "rb") as f:
        raw_data = pickle.load(f)
        for pair, dist_data in raw_data.items():
            e1 = pair[0].replace(" ", "_")
            e2 = pair[1].replace(" ", "_")
            total = sum(dist_data.values())

            data[(e1, e2)] = {"years": dist_data, "total": total}
            data[(e1, e1)] = {"years": dist_data, "total": total}

    return data

def harm_mean(x, y):
    try:
        return 2 * x * y / (x + y)
    except ZeroDivisionError:
        return 0


def process_hypotheses(hyp_list, ent_type, conns_train, ents_sentences, 
                       pair_counts, hyp_labels, hyp_scores, cos_sim, graph_embs):
    header = ["s", "p", "o", "type_s", "type_o", "scicheck_score", "num_connections_train_subject", "num_connections_train_object",
              "freq_abstracts_subject", "freq_abstracts_object", "harmomic_mean_frequences",
              "freq_abstracts_s&o", "max_cos_sim", "graph_emb_similarity", "word_emb_similarity", "label"]

    with open("hypotheses_processed.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for s, p, o in tqdm(hyp_list):
            scicheck_score = hyp_scores[(s,p,o)]
            if s == o or scicheck_score < 0.5:
                continue

            line = [s, p, o]
            line.append(ent_type.get(s, "?"))
            line.append(ent_type.get(o, "?"))
            line.append(scicheck_score)
            line.append(conns_train.get(s, 0))
            line.append(conns_train.get(o, 0))
            line.append(ents_sentences.get(s, {}).get("total", 0))
            line.append(ents_sentences.get(o, {}).get("total", 0))
            line.append(harm_mean(line[-1], line[-2]))
            line.append(pair_counts.get((s, o), {}).get("total", 0))
            line.append(cos_sim.get((s,p,o), 0.0))
            line.append(graph_emb_similarity(s, o, graph_embs))
            line.append(word_emb_similarity(s, o))
            line.append(hyp_labels[(s,p,o)])

            f.write(",".join(str(x) for x in line) + "\n")

def main():
    print("Loading train data...")
    conns_train, _ = load_train()

    print("Loading entity types...")
    ent_type = load_types()

    print("Loading hypothesis data... ", end="")
    hyp_list, hyp_scores, hyp_labels = load_hypotheses()
    print(len(hyp_list), "found.")

    print("Loading graph embeddings...")
    graph_embs = load_graph_embeddings()

    print("Loading entity and pair counts...")
    ents_sentences = load_entity_sentences()
    pair_counts = load_entity_pairs()

    print("Loading cosine similarity data...")
    cos_sim = load_max_cos_similarity()

    print("Processing hypotheses")
    process_hypotheses(hyp_list, ent_type, conns_train, 
                       ents_sentences, pair_counts, 
                       hyp_labels, hyp_scores, cos_sim,
                       graph_embs)

if __name__ == "__main__":
    main()
