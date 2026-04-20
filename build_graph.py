import os
import json
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import geohash

from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import cosine_similarity

from param_parser import parameter_parser


# ------------------------------------------------------------
# JSON Loader
# ------------------------------------------------------------

def load_line_json(filepath):
    """Load JSON file where each line is a JSON object"""
    data = {}

    with open(filepath, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:
                obj = json.loads(line)
                data.update(obj)

            except json.JSONDecodeError:
                print("Warning: invalid json line skipped")

    return data


# ------------------------------------------------------------
# Geohash
# ------------------------------------------------------------

def geohash_add2_df(df, precision=6):

    df["geohash"] = [
        geohash.encode(lat, lon, precision)
        for lat, lon in zip(df.latitude, df.longitude)
    ]

    return df


# ------------------------------------------------------------
# Trajectory Graph
# ------------------------------------------------------------

def build_global_POI_checkin_graph(df, exclude_user=None):

    G = nx.DiGraph()

    grouped = df.groupby("user_id")

    users = list(grouped.groups.keys())

    if exclude_user in users:
        users.remove(exclude_user)

    for user_id in tqdm(users):

        user_df = grouped.get_group(user_id)

        prev_poi = None
        prev_traj = None

        for row in user_df.itertuples():

            poi = row.POI_id

            if poi not in G:

                G.add_node(
                    poi,
                    checkin_cnt=1,
                    poi_catid=row.POI_catid,
                    poi_catid_code=row.POI_catid_code,
                    poi_catname=row.POI_catname,
                    latitude=row.latitude,
                    longitude=row.longitude,
                )

            else:

                G.nodes[poi]["checkin_cnt"] += 1

            if prev_poi is not None and prev_traj == row.trajectory_id:

                if G.has_edge(prev_poi, poi):
                    G[prev_poi][poi]["weight"] += 1
                else:
                    G.add_edge(prev_poi, poi, weight=1)

            prev_poi = poi
            prev_traj = row.trajectory_id

    return G


def apply_tfidf_weighting(G):
    """
    对转移图应用TF-IDF加权
    TF = count(i→j) / Σ_k count(i→k)
    IDF = log(|P| / n_j)
    weight = TF * IDF
    """
    num_pois = G.number_of_nodes()

    # 计算每个POI被转移到的次数（用于IDF）
    in_degree_count = {}
    for node in G.nodes():
        in_degree_count[node] = len(list(G.predecessors(node)))

    # 对每条边应用TF-IDF权重
    for src in G.nodes():
        out_edges = list(G.out_edges(src, data=True))

        if len(out_edges) == 0:
            continue

        # 计算TF：归一化转移频率
        total_out_weight = sum([edge[2]['weight'] for edge in out_edges])

        for src_node, dst_node, data in out_edges:
            tf = data['weight'] / total_out_weight

            # 计算IDF
            n_dst = max(in_degree_count[dst_node], 1)
            idf = np.log(num_pois / n_dst)

            # TF-IDF权重
            G[src_node][dst_node]['weight'] = tf * idf

    return G


# ------------------------------------------------------------
# Geo Graph
# ------------------------------------------------------------

def build_global_POI_geo_graph(df):

    G = nx.DiGraph()

    grouped = df.groupby("user_id")

    for user_id in tqdm(grouped.groups):

        user_df = grouped.get_group(user_id)

        prev_geo = None
        prev_traj = None

        for row in user_df.itertuples():

            geo = row.geohash

            if geo not in G:
                G.add_node(geo, checkin_cnt=1)
            else:
                G.nodes[geo]["checkin_cnt"] += 1

            if prev_geo is not None and prev_traj == row.trajectory_id:

                if G.has_edge(prev_geo, geo):
                    G[prev_geo][geo]["weight"] += 1
                else:
                    G.add_edge(prev_geo, geo, weight=1)

            prev_geo = geo
            prev_traj = row.trajectory_id

    return G


# ------------------------------------------------------------
# Embedding Fusion
# ------------------------------------------------------------

def fuse_embeddings(image_embeds, review_embeds, meta_embeds):

    poi_embeddings = {}

    common_pois = set(image_embeds) & set(review_embeds) & set(meta_embeds)

    for poi in common_pois:

        img = np.asarray(image_embeds[poi], dtype=np.float32)
        rev = np.asarray(review_embeds[poi], dtype=np.float32)
        meta = np.asarray(meta_embeds[poi], dtype=np.float32)

        dim = min(len(img), len(rev), len(meta))

        fused = (img[:dim] + rev[:dim] + meta[:dim]) / 3

        poi_embeddings[poi] = fused

    return poi_embeddings


# ------------------------------------------------------------
# Semantic Graph (FAST)
# ------------------------------------------------------------

def build_multimodal_semantic_graph(poi_embeddings, poi_df, distance_threshold=1.0):

    G = nx.Graph()

    coords = poi_df[["latitude", "longitude"]].values
    poi_ids = poi_df["POI_id"].values

    coords_rad = np.radians(coords)

    tree = BallTree(coords_rad, metric="haversine")

    radius = distance_threshold / 6371.0

    for i, poi in enumerate(tqdm(poi_ids)):

        G.add_node(
            poi,
            checkin_cnt=1,
            poi_catid=0,
            poi_catid_code=0,
            poi_catname="semantic",
            latitude=coords[i][0],
            longitude=coords[i][1],
        )

        ind = tree.query_radius(coords_rad[i:i + 1], r=radius)[0]

        for j in ind:

            if j <= i:
                continue

            poi_j = poi_ids[j]

            emb_i = poi_embeddings.get(poi)
            emb_j = poi_embeddings.get(poi_j)

            if emb_i is None or emb_j is None:
                continue

            sim = cosine_similarity(
                emb_i.reshape(1, -1),
                emb_j.reshape(1, -1),
            )[0][0]

            G.add_edge(poi, poi_j, weight=float(sim))

    return G


# ------------------------------------------------------------
# Save Graph
# ------------------------------------------------------------

def save_graph_to_csv(G, dst_dir, name):

    nodelist = list(G.nodes())

    A = nx.adjacency_matrix(G, nodelist=nodelist)

    np.savetxt(
        os.path.join(dst_dir, f"{name}_A.csv"),
        A.todense(),
        delimiter=",",
    )

    with open(os.path.join(dst_dir, f"{name}_X.csv"), "w", encoding="utf-8") as f:

        print(
            "node_name/poi_id\tcheckin_cnt\tpoi_catid\tpoi_catid_code\tpoi_catname\tlatitude\tlongitude",
            file=f,
        )

        for node in nodelist:

            data = G.nodes[node]

            checkin_cnt = data.get("checkin_cnt", 0)
            poi_catid = data.get("poi_catid", 0)
            poi_catid_code = data.get("poi_catid_code", 0)
            poi_catname = data.get("poi_catname", "unknown")
            lat = data.get("latitude", 0)
            lon = data.get("longitude", 0)

            print(
                f"{node}\t{checkin_cnt}\t{poi_catid}\t{poi_catid_code}\t{poi_catname}\t{lat}\t{lon}",
                file=f,
            )
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    args = parameter_parser()

    dst_dir = "dataset/NYC"

    print("Loading dataset...")

    train_df = pd.read_csv(os.path.join(dst_dir, "NYC_train.csv"))

    # --------------------------------------------------------
    # Trajectory Graph
    # --------------------------------------------------------

    print("Building POI trajectory graph...")

    G_traj = build_global_POI_checkin_graph(train_df)

    print("Applying TF-IDF weighting...")
    G_traj = apply_tfidf_weighting(G_traj)

    save_graph_to_csv(G_traj, dst_dir, "graph")


    # --------------------------------------------------------
    # Geo Graph
    # --------------------------------------------------------

    print("Building geo graph...")

    train_df = geohash_add2_df(train_df)

    G_geo = build_global_POI_geo_graph(train_df)

    save_graph_to_csv(G_geo, dst_dir, "geo_graph")


    # --------------------------------------------------------
    # Load Embeddings
    # --------------------------------------------------------

    print("Loading embeddings...")

    image_embeds = load_line_json(
        os.path.join(dst_dir, "modal_image_embedding.json")
    )

    review_embeds = load_line_json(
        os.path.join(dst_dir, "modal_review_embedding.json")
    )

    meta_embeds_raw = load_line_json(
        os.path.join(dst_dir, "modal_meta_embedding.json")
    )

    meta_embeds = {}

    for poi_id, emb in meta_embeds_raw.items():

        if isinstance(emb, list) and len(emb) == 1:
            meta_embeds[poi_id] = emb[0]
        else:
            meta_embeds[poi_id] = emb

    print("Fusing embeddings...")

    poi_embeddings = fuse_embeddings(
        image_embeds,
        review_embeds,
        meta_embeds,
    )

    print("Total POI embeddings:", len(poi_embeddings))


    # --------------------------------------------------------
    # Semantic Graph
    # --------------------------------------------------------

    print("Building semantic graph...")

    poi_df = train_df[
        ["POI_id", "latitude", "longitude"]
    ].drop_duplicates()

    G_semantic = build_multimodal_semantic_graph(
        poi_embeddings,
        poi_df,
        distance_threshold=args.semantic_graph_distance,
    )

    print(
        "Semantic graph:",
        G_semantic.number_of_nodes(),
        "nodes",
        G_semantic.number_of_edges(),
        "edges",
    )

    save_graph_to_csv(G_semantic, dst_dir, "semantic_graph")

    print("All graphs built successfully.")