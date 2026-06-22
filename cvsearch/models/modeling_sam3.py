import spacy
import re
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw
import time
import matplotlib.patches as patches
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.visualization_utils import plot_results
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage import graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
try:
    from skimage import graph
except ImportError:
    from skimage.future import graph
import gc

class sam3_inference():
    def __init__(self, model_path):
        self.model = build_sam3_image_model(checkpoint_path=model_path)
        self.model.eval()
        self.model.to("cuda")
        self.processor = Sam3Processor(self.model)
        self.transform = ComposeAPI(
                        transforms=[
                            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
                            ToTensorAPI(),
                            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        ]
                    )

        self.postprocessor = PostProcessImage(
                            max_dets_per_img=-1,       # if this number is positive, the processor will return topk. For this demo we instead limit by confidence, see below
                            iou_type="segm",           # we want masks
                            use_original_sizes_box=True,   # our boxes should be resized to the image size
                            use_original_sizes_mask=True,   # our masks should be resized to the image size
                            convert_mask_to_rle=False, # the postprocessor supports efficient conversion to RLE format. In this demo we prefer the binary format for easy plotting
                            detection_threshold=0.5,   # Only return confident detections
                            to_cpu=False,
                        )
        self.GLOBAL_COUNTER = 1

    @torch.inference_mode()
    def inference(self, image, text_prompt):
        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        return output

    def create_empty_datapoint(self):
        """ A datapoint is a single image on which we can apply several queries at once. """
        return Datapoint(find_queries=[], images=[])

    def set_image(self, datapoint, pil_image):
        """ Add the image to be processed to the datapoint """
        w, h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

        return datapoint

    def add_text_prompt(self, datapoint, text_query, current_id):
        """ Add a text query to the datapoint """
        # in this function, we require that the image is already set.
        # that's because we'll get its size to figure out what dimension to resize masks and boxes
        # In practice you're free to set any size you want, just edit the rest of the function
        assert len(datapoint.images) == 1, "please set the image first"

        w, h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],  # unused for inference
                is_exhaustive=True,  # unused for inference
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=current_id,
                    original_image_id=current_id,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
        )
        self.GLOBAL_COUNTER += 1
        return datapoint, current_id

    @torch.inference_mode()
    def batch_inference(self, image, text_prompts):
        datapoint = self.create_empty_datapoint()
        datapoint = self.set_image(datapoint, image)
        text_id = []
        for idx, text in enumerate(text_prompts, start=1):
            datapoint, t_id = self.add_text_prompt(datapoint, text, idx)
            text_id.append(t_id)

        datapoint = self.transform(datapoint)
        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

        output, backbone_out = self.model(batch)
        processed_results = self.postprocessor.process_results(output, batch.find_metadatas)

        return backbone_out, processed_results, text_id

    def plot_results(self, image, processed_result, t_id):
        plot_results(image, processed_result[t_id])

# --- Effective Rank ---
def _calc_complexity_effective_rank(features):
    """
    Effective Rank = exp(Shannon Entropy of Singular Values)
    Args:
        features: (N_subset, C)
    Returns:
        float: 0 ~ 1
    """
    N, C = features.shape
    if N <= 1 or C == 0:
        return 0.0

    centered = features - features.mean(axis=0)

    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        s_sq = s ** 2
        total_energy = np.sum(s_sq) + 1e-10
        probs = s_sq / total_energy
        valid_probs = probs[probs > 1e-10]
        if len(valid_probs) == 0:
            return 0.0
        entropy = -np.sum(valid_probs * np.log(valid_probs))
        effective_rank = np.exp(entropy)
        max_rank = min(N, C)
        if max_rank <= 0: return 0.0

        return effective_rank

    except np.linalg.LinAlgError:
        return 0.0

class ConstrainedTreeBuilder:
    def __init__(self, feature_map, n_atoms=400, pos_weight=2.0, split_threshold=0.3, keep_threshold=0.05, lazy_base=0.4, lazy_bonus=0.6, decay_factor=0.95, use_local_normalization=True,
                 use_silhouette_score=True):
        if isinstance(feature_map, torch.Tensor):
            self.feat = feature_map.detach().cpu().numpy()
        else:
            self.feat = feature_map

        self.C, self.H, self.W = self.feat.shape
        self.n_atoms = n_atoms
        self.pos_weight = pos_weight
        self.split_threshold = split_threshold
        self.keep_threshold = keep_threshold
        self.lazy_base = lazy_base
        self.lazy_bonus = lazy_bonus
        self.decay_factor = decay_factor
        self.use_local_normalization = use_local_normalization
        self.use_silhouette_score = use_silhouette_score
        self.node_registry = {}
        self.atom_labels, self.atom_features, self.atom_bboxes, self.adj_matrix = self._generate_atoms_and_graph()

    def _generate_atoms_and_graph(self):
        feat_tr = self.feat.transpose(1, 2, 0)
        feat_min, feat_max = feat_tr.min(), feat_tr.max()
        feat_norm = (feat_tr - feat_min) / (feat_max - feat_min + 1e-6)
        #SLIC
        atom_map = slic(feat_norm, n_segments=self.n_atoms, compactness=20, start_label=0, channel_axis=2)
        unique_labels = np.unique(atom_map)
        n_actual = len(unique_labels)
        semantic_features = np.zeros((n_actual, self.C), dtype=np.float32)
        spatial_features = np.zeros((n_actual, 2), dtype=np.float32)
        props = regionprops(atom_map + 1)
        atom_bboxes = []

        for i, prop in enumerate(props):
            y_slice, x_slice = prop.slice
            mask_local = prop.image
            feat_crop = self.feat[:, y_slice, x_slice]
            semantic_features[i] = feat_crop[:, mask_local].mean(axis=1)

            cy, cx = prop.centroid
            y_encoded = (cy / self.H) * self.pos_weight
            x_encoded = (cx / self.W) * self.pos_weight
            spatial_features[i] = [y_encoded, x_encoded]

            atom_bboxes.append(prop.bbox)

        final_features = np.concatenate([semantic_features, spatial_features], axis=1)

        # --- Construct adjacency matrix ---
        rag_img = feat_tr[:, :, :3] if self.C >= 3 else feat_tr
        rag = graph.rag_mean_color(rag_img, atom_map, mode='distance')

        # Convert to Sparse Matrix (N_atoms, N_atoms)
        adj_matrix = nx.adjacency_matrix(rag)
        return atom_map, final_features, np.array(atom_bboxes), adj_matrix

    def _calc_overlap_cost(self, child_nodes):
        if len(child_nodes) < 2: return 0.0

        boxes = [n['bbox'] for n in child_nodes]
        total_area = sum([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        total_overlap = 0.0

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ba, bb = boxes[i], boxes[j]
                iy1, ix1 = max(ba[0], bb[0]), max(ba[1], bb[1])
                iy2, ix2 = min(ba[2], bb[2]), min(ba[3], bb[3])
                inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
                total_overlap += inter

        if total_area == 0: return 0
        return total_overlap / total_area

    def _calc_region_complexity(self, atom_indices):
        if len(atom_indices) <= 1:
            return 0.0

        # semantic features (N_subset, C)
        features = self.atom_features[atom_indices, :self.C]
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        feats_norm = features / norm
        mean_feat = feats_norm.mean(axis=0)
        mean_feat_norm = mean_feat / (np.linalg.norm(mean_feat) + 1e-6)

        # cosine similarity
        # shapes: (N, C) @ (C,) -> (N,)
        cosine_sims = feats_norm @ mean_feat_norm
        avg_sim = cosine_sims.mean()
        # Convert to complexity score
        score = max(0.0, 1.0 - avg_sim)

        avg_magnitude = norm.mean()
        score *= np.log1p(avg_magnitude)

        return score

    def build_tree(self, max_depth=3, min_splits=4, max_splits=8):
        self.node_registry = {}
        all_indices = np.arange(len(self.atom_features))
        # Calculate the complexity of the global image
        global_complexity = self._calc_region_complexity(all_indices)
        print(f"Global Image Complexity Score: {global_complexity:.4f}")

        root_node = {
            "depth": 0,
            "atom_indices": all_indices,
            "bbox": (0, 0, self.H, self.W),
            "children": [],
            "split_k": 1,
            "complexity": global_complexity,
            "relative_score": 1.0,
            "node_id": "0",
            "prior_prob": 1.0, #root node 1.0
            "parent": None
        }
        self.node_registry["0"] = root_node
        print("Building Constrained Tree with Semantic Pruning...")
        self._recursive_build(root_node, max_depth, min_splits, max_splits)
        return root_node

    def _recursive_build(self, parent_node, max_depth, min_splits, max_splits):
        if parent_node["depth"] >= max_depth:
            return

        indices = parent_node["atom_indices"]
        effective_min_k = min_splits
        if len(indices) < max(min_splits, 2):
            return

        # Semantic Pruning
        if parent_node["depth"] >= 1:
            depth_decay = 0.8 ** (parent_node["depth"] - 1)
            current_threshold = self.split_threshold * depth_decay

            complexity = parent_node.get("complexity", 0)
            if complexity < current_threshold:
                return

        raw_sub_features = self.atom_features[indices]
        semantic_raw = raw_sub_features[:, :self.C]
        if self.use_local_normalization:
            scaler = StandardScaler()
            enhanced_semantic = scaler.fit_transform(semantic_raw)
        else:
            enhanced_semantic = semantic_raw

        spatial_feats = raw_sub_features[:, self.C:]
        spatial_scale = 1.0 / (parent_node["depth"] + 1)
        sub_features_for_clustering = np.concatenate([enhanced_semantic, spatial_feats * spatial_scale], axis=1)

        sub_connectivity = self.adj_matrix[indices, :][:, indices]

        best_score = -float('inf')
        best_children = []
        best_k = effective_min_k
        best_labels = None
        limit_k = min(max_splits, len(indices))
        if limit_k < effective_min_k: return

        for k in range(effective_min_k, limit_k + 1):
            try:
                model = AgglomerativeClustering(
                    n_clusters=k,
                    connectivity=sub_connectivity,
                    linkage='ward'
                )
                labels = model.fit_predict(sub_features_for_clustering)

                if self.use_silhouette_score:
                    if k > 1 and len(indices) > k:
                        sil_score = silhouette_score(enhanced_semantic, labels)
                    else:
                        sil_score = -1.0
                else:
                    sil_score = 0.0

                # Calculate BBox overlap cost
                overlap_cost = self._calc_overlap_cost_for_labels(indices, labels, k)
                # --- Comprehensive scoring formula ---
                combined_score = sil_score - (1.5 * overlap_cost)
                if combined_score > best_score:
                    best_score = combined_score
                    best_k = k
                    best_labels = labels

            except Exception as e:
                continue

        if best_score == -float('inf'): return
        best_children = []
        for lbl in range(best_k):
            child_indices_local = np.where(best_labels == lbl)[0]
            if len(child_indices_local) == 0: continue

            child_atom_indices = indices[child_indices_local]
            c_boxes = self.atom_bboxes[child_atom_indices]
            y1, x1 = np.min(c_boxes[:, 0]), np.min(c_boxes[:, 1])
            y2, x2 = np.max(c_boxes[:, 2]), np.max(c_boxes[:, 3])

            best_children.append({
                "atom_indices": child_atom_indices,
                "bbox": (y1, x1, y2, x2)
            })

        if not best_children: return

        # Calculate complexity and prune
        valid_children_data = []  # (child_data, complexity)
        complexities = []

        for child_data in best_children:
            child_complexity = self._calc_region_complexity(child_data["atom_indices"])
            if child_complexity < self.keep_threshold:
                continue

            valid_children_data.append((child_data, child_complexity))
            complexities.append(child_complexity)

        # All child nodes have been pruned
        if not valid_children_data:
            return

        # relative_score and prior_prob
        max_c = max(complexities)
        min_c = min(complexities)
        range_c = max_c - min_c

        parent_prob = parent_node.get("prior_prob", 1.0)
        for i, (child_data, child_complexity) in enumerate(valid_children_data):

            # --- Intra-Level Normalization)
            if range_c > 1e-6:
                relative_score = (child_complexity - min_c) / range_c
            else:
                relative_score = 1.0
            relative_score = 0.2 + 0.8 * relative_score

            # prior_prob=Parent_Prob * (Base + Bonus * Relative) * Decay
            estimated_transfer = self.lazy_base + self.lazy_bonus * relative_score
            current_prob = parent_prob * estimated_transfer * self.decay_factor
            current_node_id = f"{parent_node['node_id']}-{i}"

            child_node = {
                "depth": parent_node["depth"] + 1,
                "atom_indices": child_data["atom_indices"],
                "bbox": child_data["bbox"],
                "children": [],
                "split_k": best_k,
                "complexity": child_complexity,
                "relative_score": relative_score,
                "prior_prob": current_prob,
                "node_id": current_node_id,
                "parent": parent_node
            }
            parent_node["children"].append(child_node)

            self.node_registry[current_node_id] = child_node
            self._recursive_build(child_node, max_depth, min_splits, max_splits)

    def _calc_overlap_cost_for_labels(self, indices, labels, k):
        temp_children = []
        for lbl in range(k):
            child_indices_local = np.where(labels == lbl)[0]
            if len(child_indices_local) == 0: continue
            child_atom_indices = indices[child_indices_local]
            c_boxes = self.atom_bboxes[child_atom_indices]
            y1, x1 = np.min(c_boxes[:, 0]), np.min(c_boxes[:, 1])
            y2, x2 = np.max(c_boxes[:, 2]), np.max(c_boxes[:, 3])
            temp_children.append({"bbox": (y1, x1, y2, x2)})
        return self._calc_overlap_cost(temp_children)

    def get_flattened_nodes(self, tree_root):
        all_nodes = []

        def _traverse(node):
            node_info = {
                'id': node['node_id'],
                'depth': node['depth'],
                'prob': node['prior_prob'],
                'bbox': node['bbox'],
                'relative_score': node['relative_score']
            }
            all_nodes.append(node_info)
            for child in node.get('children', []):
                _traverse(child)

        _traverse(tree_root)
        return sorted(all_nodes, key=lambda x: x['prob'], reverse=True)

    def get_node_by_id(self, node_id):

        return self.node_registry.get(node_id)

class TreeVisualizer:
    def __init__(self, tree_root, image, feature_map_shape):
        self.root = tree_root
        self.image = image
        self.img_w, self.img_h = self.image.size

        if len(feature_map_shape) == 3:
            _, self.feat_h, self.feat_w = feature_map_shape
        else:
            self.feat_h, self.feat_w = feature_map_shape

        # Scale Factor
        self.scale_x = self.img_w / self.feat_w
        self.scale_y = self.img_h / self.feat_h

        self.max_tree_depth = self._get_max_depth(self.root)
        print(f"Tree Max Depth: {self.max_tree_depth}")

        print(f"Original Image: {self.img_w}x{self.img_h}")
        print(f"Feature Map: {self.feat_w}x{self.feat_h}")
        print(f"Scale Factor: X={self.scale_x:.2f}, Y={self.scale_y:.2f}")

    def _get_max_depth(self, node):
        if not node['children']:
            return node['depth']
        return max(self._get_max_depth(child) for child in node['children'])

    def _convert_bbox(self, feat_bbox):
        """
        BBox (y1, x1, y2, x2) to BBox (x1, y1, x2, y2)
        PIL crop (left, upper, right, lower)->(x1, y1, x2, y2)
        """
        y1, x1, y2, x2 = feat_bbox

        orig_x1 = int(x1 * self.scale_x)
        orig_y1 = int(y1 * self.scale_y)
        orig_x2 = int(x2 * self.scale_x)
        orig_y2 = int(y2 * self.scale_y)

        orig_x1 = max(0, orig_x1)
        orig_y1 = max(0, orig_y1)
        orig_x2 = min(self.img_w, orig_x2)
        orig_y2 = min(self.img_h, orig_y2)

        return (orig_x1, orig_y1, orig_x2, orig_y2)

    def _collect_nodes_at_depth(self, current_node, target_depth, collected_nodes):
        if current_node['depth'] == target_depth:
            collected_nodes.append(current_node)
            return

        if current_node['depth'] < target_depth:
            for child in current_node['children']:
                self._collect_nodes_at_depth(child, target_depth, collected_nodes)

    def visualize_layer(self, target_depth=0):
        target_nodes = []
        self._collect_nodes_at_depth(self.root, target_depth, target_nodes)

        if not target_nodes:
            print(f"No nodes found at Depth {target_depth}")
            return

        print(f"\n=== Visualizing Depth {target_depth} -> Depth {target_depth + 1} ===")
        print(f"Found {len(target_nodes)} parent nodes at Depth {target_depth}.")

        for i, parent_node in enumerate(target_nodes):
            bbox_parent = self._convert_bbox(parent_node['bbox'])
            children = parent_node['children']
            n_children = len(children)
            complexity = parent_node.get('complexity', 'N/A')
            try:
                comp_str = f"{float(complexity):.2f}"
            except (ValueError, TypeError):
                comp_str = str(complexity)
            relative_score = parent_node.get('relative_score', 'N/A')
            try:
                rela_str = f"{float(relative_score):.2f}"
            except (ValueError, TypeError):
                rela_str = str(relative_score)
            node_id = parent_node.get("node_id", 'N/A')
            crop_parent = self.image.crop(bbox_parent)

            cols = max(2, n_children)
            fig = plt.figure(figsize=(15, 4))

            ax_main = plt.subplot2grid((1, cols + 2), (0, 0), colspan=2)

            crop_parent_draw = crop_parent.copy()
            draw = ImageDraw.Draw(crop_parent_draw)
            offset_x, offset_y = bbox_parent[0], bbox_parent[1]

            for child in children:
                cx1, cy1, cx2, cy2 = self._convert_bbox(child['bbox'])
                local_box = (cx1 - offset_x, cy1 - offset_y, cx2 - offset_x, cy2 - offset_y)
                draw.rectangle(local_box, outline="yellow", width=3)

            ax_main.imshow(crop_parent_draw)
            ax_main.set_title(f"{node_id} (Score: {comp_str} Relative Score: {rela_str})\nChildren: {n_children}",
                              fontweight='bold', color='darkblue')
            ax_main.axis('off')

            if n_children > 0:
                for j, child in enumerate(children):
                    ax_sub = plt.subplot2grid((1, cols + 2), (0, j + 2))

                    c_bbox = self._convert_bbox(child['bbox'])
                    c_crop = self.image.crop(c_bbox)

                    c_comp = child.get('complexity', 'N/A')
                    try:
                        c_comp_str = f"{float(c_comp):.2f}"
                    except (ValueError, TypeError):
                        c_comp_str = str(c_comp)

                    c_score = child.get('relative_score', 'N/A')
                    try:
                        c_score_str = f"{float(c_score):.2f}"
                    except (ValueError, TypeError):
                        c_score_str = str(c_score)

                    c_node_id = child.get('node_id', 'N/A')

                    ax_sub.imshow(c_crop)
                    ax_sub.set_title(f"{c_node_id}\nScore: {c_comp_str} Relative Score: {c_score_str}", fontsize=9)
                    ax_sub.axis('off')
            else:
                msg = "Leaf Node\n(No further split)"
                plt.text(0.6, 0.5, msg, transform=fig.transFigure, fontsize=12, color='gray')

            plt.tight_layout()
            plt.show()

    def visualize_all_levels(self):
        print(f"Starting Full Hierarchy Visualization (Max Depth: {self.max_tree_depth})")
        for d in range(self.max_tree_depth):
            self.visualize_layer(target_depth=d)

    def visualize_hierarchy(self):
        level1_nodes = self.root['children']

        if not level1_nodes:
            print("Tree has no children at Depth 1.")
            return

        n_l1 = len(level1_nodes)
        print(f"Found {n_l1} blocks at Depth 1.")

        for i, node_l1 in enumerate(level1_nodes):
            bbox_l1 = self._convert_bbox(node_l1['bbox'])
            crop_l1 = self.image.crop(bbox_l1)

            level2_nodes = node_l1['children']
            n_l2 = len(level2_nodes)

            cols = max(2, n_l2)
            fig = plt.figure(figsize=(15, 4))
            ax_main = plt.subplot2grid((1, cols + 2), (0, 0), colspan=2)

            crop_l1_draw = crop_l1.copy()
            draw = ImageDraw.Draw(crop_l1_draw)

            offset_x, offset_y = bbox_l1[0], bbox_l1[1]

            for node_l2 in level2_nodes:
                bx1, by1, bx2, by2 = self._convert_bbox(node_l2['bbox'])
                local_box = (bx1 - offset_x, by1 - offset_y, bx2 - offset_x, by2 - offset_y)
                draw.rectangle(local_box, outline="yellow", width=3)

            ax_main.imshow(crop_l1_draw)
            ax_main.set_title(f"Depth 1 - Block {i}\n(Contains {n_l2} sub-blocks) info {node_l1['complexity']}",
                              fontweight='bold')
            ax_main.axis('off')

            if n_l2 > 0:
                for j, node_l2 in enumerate(level2_nodes):
                    ax_sub = plt.subplot2grid((1, cols + 2), (0, j + 2))
                    bbox_l2 = self._convert_bbox(node_l2['bbox'])
                    crop_l2 = self.image.crop(bbox_l2)
                    ax_sub.imshow(crop_l2)
                    ax_sub.set_title(f"D2 - Sub {j} info {node_l2['complexity']}", fontsize=9)
                    ax_sub.axis('off')
            else:
                plt.text(0.5, 0.5, "No Leaf Nodes", transform=fig.transFigure)

            plt.tight_layout()
            plt.show()

    # ==========================================
    # PCA
    # ==========================================
    def visualize_pca(self, feat):
        # (N_pixels, C) -> (5184, 256)
        C, H, W = feat.shape
        reshaped_feat = feat.reshape(C, -1).T

        # PCA = 3
        pca = PCA(n_components=3)
        pca_feat = pca.fit_transform(reshaped_feat)
        feat_min = pca_feat.min(axis=0)
        feat_max = pca_feat.max(axis=0)
        pca_feat = (pca_feat - feat_min) / (feat_max - feat_min)

        # (H, W, 3)
        pca_img = pca_feat.reshape(H, W, 3)

        plt.figure(figsize=(5, 5))
        plt.title("PCA Feature Visualization (RGB)")
        plt.imshow(pca_img)
        plt.axis('off')
        plt.show()

    def visualize_l2_norm(self, feat):

        l2_map = np.linalg.norm(feat, axis=0)

        plt.figure(figsize=(5, 5))
        plt.title("L2 Norm Heatmap (Feature Magnitude)")
        plt.imshow(l2_map, cmap='inferno')
        plt.colorbar(label='Feature Norm')
        plt.axis('off')
        plt.show()

    def visualize_adaptive_by_depth(self, root_node, ax, target_depth=1):
        nodes_at_depth = []

        def collect_recursive(node):
            if node['depth'] == target_depth:
                nodes_at_depth.append(node)
                return

            if node['depth'] < target_depth:
                for child in node['children']:
                    collect_recursive(child)

        collect_recursive(root_node)
        cmap = plt.get_cmap('tab20')
        n_colors = 20
        count = len(nodes_at_depth)
        ax.set_title(f"Depth {target_depth} (Total Blocks: {count})")

        for i, node in enumerate(nodes_at_depth):
            y1, x1, y2, x2 = node['bbox']
            color_rgba = cmap(i % n_colors)
            color_rgb = color_rgba[:3]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                     edgecolor=color_rgb,
                                     facecolor=color_rgb + (0.2,),
                                     label=f'Block {i}')
            ax.add_patch(rect)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(cx, cy, f"{i}", color='white', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    bbox=dict(facecolor=color_rgb, alpha=0.6, edgecolor='none', pad=1))

    def visualize_all_depth(self, tree, depth):
        fig, axes = plt.subplots(1, depth, figsize=(depth * 6, 6))
        img_shape = (72, 72)

        for target_depth in range(depth):
            # Depth
            axes[target_depth].imshow(np.zeros(img_shape), cmap='gray')
            self.visualize_adaptive_by_depth(tree, axes[target_depth], target_depth=target_depth + 1)

        plt.tight_layout()
        plt.show()

    def visualize_subtrees_individually(self, root_node, img_shape=(72, 72)):
        subtrees = root_node['children']
        n_subtrees = len(subtrees)

        if n_subtrees == 0:
            print("Root has no children to visualize.")
            return

        fig, axes = plt.subplots(1, n_subtrees, figsize=(5 * n_subtrees, 5))
        if n_subtrees == 1: axes = [axes]
        cmap = plt.get_cmap('tab10')

        def draw_recursive(node, ax, base_depth):
            y1, x1, y2, x2 = node['bbox']
            rel_depth = node['depth'] - base_depth
            color = cmap(node['depth'] % 10)
            linewidth = max(1, 4 - rel_depth)

            alpha = max(0.1, 0.4 - rel_depth * 0.1)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=linewidth,
                                     edgecolor=color,
                                     facecolor=color[:3] + (alpha,),
                                     linestyle='-' if rel_depth == 0 else '--',
                                     label=f'D{node["depth"]}' if rel_depth == 0 else None)
            ax.add_patch(rect)
            if rel_depth == 0:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(cx, cy, f"Root\n{node['depth']}", color='white', ha='center', va='center',
                        fontweight='bold')

            for child in node['children']:
                draw_recursive(child, ax, base_depth)

        for i, subtree_root in enumerate(subtrees):
            ax = axes[i]
            ax.imshow(np.zeros(img_shape), cmap='gray')

            def count_nodes(n): return 1 + sum(count_nodes(c) for c in n['children'])

            total_nodes = count_nodes(subtree_root)

            ax.set_title(f"Subtree {i}\n(Contains {total_nodes} nodes)")
            draw_recursive(subtree_root, ax, base_depth=subtree_root['depth'])
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_mask(self, image, text_prompt, masks, boxes, scores, left, top):
        original_image = np.array(image)
        num_masks = masks.shape[0]

        plt.figure(figsize=(12, 8))
        plt.imshow(original_image)
        plt.axis('off')
        plt.title(f"Text Prompt: '{text_prompt}' | Detected {num_masks} object(s)")

        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_masks))

        for i in range(num_masks):
            color = colors[i][:3]
            x1, y1, x2, y2 = boxes[i]
            x1, x2 = x1 + left, x2 + left
            y1, y2 = y1 + top, y2 + top
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x1, y1 - 5, f"{scores[i]:.2f}", color=color, fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.5, pad=1))

        plt.tight_layout()
        plt.show()

def calculate_and_sort_prior_probs(
        tree_root,
        lazy_base: float = 0.4,
        lazy_bonus: float = 0.6,
        decay_factor: float = 0.95
):
    """
    Args:
        tree_root (dict)
        lazy_base (float)
        lazy_bonus (float)
        decay_factor (float)
    Returns:
        sorted_nodes (list): depth, id, prob, bbox
    """
    # {'id': str, 'depth': int, 'prob': float, 'bbox': tuple, 'relative_score': float}
    all_nodes_flat = []

    def _traverse(node, parent_prob):
        rel_score = node.get('relative_score', 1.0)
        node_id = node.get('node_id', 'root')
        depth = node['depth']
        if depth == 0:
            current_prob = 1.0
        else:
            estimated_transfer = lazy_base + lazy_bonus * rel_score
            current_prob = parent_prob * estimated_transfer * decay_factor

        node['prior_prob'] = current_prob
        node_info = {
            'id': node_id,
            'depth': depth,
            'prob': current_prob,
            'bbox': node['bbox'],  # (y1, x1, y2, x2)
            'relative_score': rel_score,
        }
        all_nodes_flat.append(node_info)
        for child in node.get('children', []):
            _traverse(child, current_prob)

    _traverse(tree_root, 1.0)
    sorted_nodes = sorted(all_nodes_flat, key=lambda x: x['prob'], reverse=True)

    return sorted_nodes, all_nodes_flat

def visualize_tree_crops_separate_figures(
        image_pil,
        all_nodes_flat,
        max_cols=6,
        feature_map_shape=(72, 72)
):
    """
    Args:
        image_pil (PIL.Image)
        all_nodes_flat (list)
        max_cols (int)
        feature_map_shape (tuple)
    """
    orig_w, orig_h = image_pil.size
    feat_h, feat_w = feature_map_shape

    scale_y = orig_h / feat_h
    scale_x = orig_w / feat_w

    print(f"--- Coordinate Transform Info ---")
    print(f"Original Image: {orig_w}x{orig_h}")
    print(f"Feature Map:    {feat_w}x{feat_h}")
    print(f"Scale Factors:  X={scale_x:.4f}, Y={scale_y:.4f}")

    max_depth = max(n['depth'] for n in all_nodes_flat)
    layers = {d: [] for d in range(max_depth + 1)}
    for node in all_nodes_flat:
        layers[node['depth']].append(node)

    for d in range(max_depth + 1):
        current_layer_nodes = layers[d]
        current_layer_nodes.sort(key=lambda x: x['prob'], reverse=True)
        n_nodes = len(current_layer_nodes)
        if n_nodes == 0:
            continue

        ncols = min(n_nodes, max_cols)
        nrows = math.ceil(n_nodes / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3.4 * nrows))
        fig.suptitle(f"=== Depth {d} (Total Nodes: {n_nodes}) ===", fontsize=16, fontweight='bold', y=0.98)

        if n_nodes == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()

        for i, ax in enumerate(axes_flat):
            if i < n_nodes:
                node = current_layer_nodes[i]
                fy1, fx1, fy2, fx2 = node['bbox']
                oy1 = int(fy1 * scale_y)
                ox1 = int(fx1 * scale_x)
                oy2 = int(fy2 * scale_y)
                ox2 = int(fx2 * scale_x)
                crop_box = (
                    max(0, ox1),
                    max(0, oy1),
                    min(orig_w, ox2),
                    min(orig_h, oy2)
                )

                if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                    patch = image_pil.crop(crop_box)
                    ax.imshow(patch)
                else:
                    ax.text(0.5, 0.5, "Region Too Small", ha='center', va='center')

                title_str = f"ID:{node['id']} P:{node['prob']:.2f}"
                if 'relative_score' in node:
                    title_str += f" R:{node['relative_score']:.2f}"

                ax.set_title(title_str, fontsize=10, color='darkblue')
                is_high_prob = node['prob'] > 0.5
                for spine in ax.spines.values():
                    spine.set_edgecolor('red' if is_high_prob else '#CCCCCC')
                    spine.set_linewidth(2.5 if is_high_prob else 1)

                ax.set_xticks([])
                ax.set_yticks([])

            else:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def print_nodes_by_priority(sorted_nodes, top_k=20):
    print(f"\n{'=' * 20} Top {top_k if top_k else 'All'} Nodes by Priority {'=' * 20}")
    print(
        f"{'Rank':<6} | {'Node ID':<10} | {'Depth':<6} | {'Prob':<8} | {'Rel Score':<10} | {'BBox (y1,x1,y2,x2)':<20}")
    print("-" * 70)

    for i, node in enumerate(sorted_nodes):
        if top_k is not None and i >= top_k:
            break

        print(
            f"{i:<6} | {node['id']:<10} | {node['depth']:<6} | {node['prob']:.4f}   | {node['relative_score']:.4f}     | {str(node['bbox'])}")
    print("-" * 70)

def print_tree_structure(node):
    if node['depth'] == 0:
        print(f"\n{'=' * 20} Tree Structure Hierarchy {'=' * 20}")

    indent = "    " * node['depth']
    branch_symbol = "└── " if node['depth'] > 0 else ""

    prob_str = f"(Prob: {node.get('prior_prob', 0):.4f})" if 'prior_prob' in node else ""

    print(f"{indent}{branch_symbol}[{node.get('node_id', 'N/A')}] {prob_str}")

    for child in node.get('children', []):
        print_tree_structure(child)

def extract_visual_objects(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    stop_nouns = set([
        "color", "position", "size", "shape", "texture", "material", "what", "where", "kind", "type",
        "side", "corner", "part", "surface", "area", "region", "level",
        "picture", "image", "photo", "scene", "background"
    ])

    def clean_chunk(text_span):
        return re.sub(r'^(the|a|an|this|that|these|those)\s+', '', text_span, flags=re.IGNORECASE).strip()

    def get_smart_expanded_span(chunk):
        root = chunk.root
        min_i = chunk.start
        max_i = chunk.end - 1

        def traverse(node):
            nonlocal max_i
            for child in node.rights:
                if child.dep_ in ['prep', 'acl', 'pobj', 'compound', 'amod', 'relcl']:
                    is_blocked = False
                    if child.dep_ == 'prep':
                        for grandchild in child.rights:
                            if grandchild.dep_ == 'pobj' and grandchild.lemma_.lower() in stop_nouns:
                                is_blocked = True
                                break

                    if not is_blocked:
                        if child.i > max_i:
                            max_i = child.i
                        traverse(child)

        traverse(root)
        return doc[min_i: max_i + 1]

    candidates = []
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ == 'PRON' or root.lemma_.lower() in stop_nouns:
            continue

        head = root.head
        condition_strict = (
                head.text in ["of", "to", "with", "from", "in", "on", "at"] or
                root.dep_ in ["dobj", "nsubj", "nsubjpass", "ROOT", "attr"]
        )
        condition_supplement = (
                root.dep_ == "conj" or
                root.dep_ == "appos" or
                (root.dep_ == "pobj" and head.dep_ != "prep")
        )

        if condition_strict or condition_supplement:
            expanded_span = get_smart_expanded_span(chunk)
            clean_text = clean_chunk(expanded_span.text)

            if clean_text:
                candidates.append({
                    "text": clean_text,
                    "root_idx": root.i,
                    "start": expanded_span.start
                })

    final_candidates = []
    candidates.sort(key=lambda x: len(x['text']), reverse=True)

    for cand in candidates:
        current_text = cand['text']
        is_contained = False
        for kept in final_candidates:
            if current_text in kept['text']:
                is_contained = True
                break
        if not is_contained:
            final_candidates.append(cand)

    final_candidates = final_candidates[:3]
    final_candidates.sort(key=lambda x: x['start'])

    objects = [c['text'] for c in final_candidates]

    if not objects:
        return [text.strip()]

    return objects

def crop_image_by_node(
        image_pil: Image.Image,
        node: dict,
        feature_map_shape: tuple = (72, 72)
):
    orig_w, orig_h = image_pil.size
    feat_h, feat_w = feature_map_shape

    scale_y = orig_h / feat_h
    scale_x = orig_w / feat_w

    if isinstance(node, dict):
        bbox = node['bbox']
    elif hasattr(node, 'bbox'):
        bbox = node.bbox
    elif hasattr(node, 'state') and hasattr(node.state, 'bbox'):
        bbox = node.state.bbox
    else:
        raise ValueError("Provided node does not contain valid bbox information.")

    fy1, fx1, fy2, fx2 = bbox
    oy1 = int(fy1 * scale_y)
    ox1 = int(fx1 * scale_x)
    oy2 = int(fy2 * scale_y)
    ox2 = int(fx2 * scale_x)

    crop_box = (
        max(0, ox1),  # left
        max(0, oy1),  # top
        min(orig_w, ox2),  # right
        min(orig_h, oy2)  # bottom
    )

    if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
        patch = image_pil.crop(crop_box)
        return patch, crop_box
    else:
        print(f"Warning: Invalid crop area for node {node.get('id', 'unknown')}: {crop_box}")
        return None, None

def process_sam_result(target_id, processed_results):
    sam_success_flags = []
    sam_bboxes = []
    for t_id in target_id:
        boxes = processed_results[t_id]["boxes"].cpu().numpy()
        if boxes.size > 0 and boxes.ndim > 1 and boxes.shape[1] >= 4:
            min_xy = np.min(boxes[:, :2], axis=0)  # [min_x, min_y]
            max_xy = np.max(boxes[:, 2:4], axis=0)  # [max_x, max_y]
            final_bbox = [int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1])]
            sam_bboxes.append(final_bbox)
            sam_success_flags.append(1)
        else:
            sam_success_flags.append(0)
            sam_bboxes.append([])

    return sam_success_flags, sam_bboxes

def plot_mask(visualizer, image, text_target, target_id, processed_results, left=0, top=0):
    for t_id in target_id:
        masks = processed_results[t_id]['masks'].cpu()
        boxes = processed_results[t_id]["boxes"].cpu().numpy()
        scores = processed_results[t_id]["scores"].cpu().numpy()
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # [N, H, W]
        else:
            masks = masks[None, :, :] if masks.ndim == 2 else masks

        visualizer.visualize_mask(image, text_target[t_id - 1], masks, boxes, scores, left, top)

if __name__ == "__main__":
    model_path = "models/facebook/sam3/sam3.pt"
    image_path = "datasets/vstar/direct_attributes/sa_19272.jpg"
    image = Image.open(image_path).convert("RGB")
    text = "cyclist's bag"
    text_target = ["woman"]  # extract_visual_objects(text)#["blue backpack", "airplane"]
    print(text_target)
    sam3 = sam3_inference(model_path=model_path)
    ##### batch inference ####
    with torch.no_grad():
        backbone_out, processed_results, target_id = sam3.batch_inference(image, text_target)
    image_features_batch = backbone_out['vision_features']
    print(image_features_batch.shape)
    # data preprocessing
    if isinstance(image_features_batch, torch.Tensor):
        feat = image_features_batch.detach().cpu().numpy()
    else:
        feat = image_features_batch
    feat = feat.squeeze(0)  # batch -> (256, 72, 72), C,H,W
    del backbone_out
    del image_features_batch
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared after 1st inference.")
    #### Building a semantic tree
    tree_depth = 3
    t0 = time.time()
    builder = ConstrainedTreeBuilder(feature_map=feat, n_atoms=600, pos_weight=3.5, split_threshold=0.3,
                                     keep_threshold=0.15, use_local_normalization=True, use_silhouette_score=True)
    tree = builder.build_tree(max_depth=tree_depth, min_splits=4, max_splits=8)
    t1 = time.time()
    print(f"Tree building time:{t1 - t0}")
    sam_success_flags, sam_bboxes = process_sam_result(target_id, processed_results)
    print(processed_results)
    print(target_id)
    print(sam_success_flags)
    print(sam_bboxes)
    feat_shape = feat.shape
    visualizer = TreeVisualizer(tree, image, feat_shape)
    ### feature map PCA-3
    visualizer.visualize_pca(feat)
    sorted_list, all_nodes_flat = calculate_and_sort_prior_probs(tree)
    visualize_tree_crops_separate_figures(
        image,
        all_nodes_flat,
        max_cols=6,
        feature_map_shape=(72, 72)
    )
    ###plot mask
    plot_mask(visualizer, image, text_target, target_id, processed_results)
    #######second inference
    target_node_id = "0-3"
    target_node = builder.get_node_by_id(target_node_id)
    if target_node:
        cropped_image, cropped_bbox = crop_image_by_node(image, target_node, feature_map_shape=(72, 72))
        left, top = cropped_bbox[0], cropped_bbox[1]
        with torch.no_grad():
            backbone_out_sub, processed_results_sub, target_id_sub = sam3.batch_inference(cropped_image,
                                                                                          text_target)

        print(processed_results_sub)
        plot_mask(visualizer, image, text_target, target_id_sub, processed_results_sub, left, top)

        image_features_batch_sub = backbone_out_sub['vision_features']
        print(image_features_batch_sub.shape)

        if isinstance(image_features_batch_sub, torch.Tensor):
            feat_sub = image_features_batch_sub.detach().cpu().numpy()
        else:
            feat_sub = image_features_batch_sub
        feat_sub = feat_sub.squeeze(0)

        del backbone_out_sub
        del image_features_batch_sub

        gc.collect()
        torch.cuda.empty_cache()
        print("GPU memory cleared after 1st inference.")

        tree_depth_sub = 2
        t0 = time.time()
        builder_sub = ConstrainedTreeBuilder(feature_map=feat_sub, n_atoms=600, pos_weight=3.5, split_threshold=0.3,
                                             keep_threshold=0.15, use_local_normalization=True,
                                             use_silhouette_score=True)
        tree_sub = builder_sub.build_tree(max_depth=tree_depth_sub, min_splits=4, max_splits=8)
        t1 = time.time()
        print(f"Tree building time:{t1 - t0}")
        sam_success_flags_sub, sam_bboxes_sub = process_sam_result(target_id_sub, processed_results_sub)
        feat_shape_sub = feat_sub.shape

        visualizer.visualize_pca(feat_sub)
        sorted_list_sub, all_nodes_flat_sub = calculate_and_sort_prior_probs(tree_sub)
        visualize_tree_crops_separate_figures(
            cropped_image,
            all_nodes_flat_sub,
            max_cols=6,
            feature_map_shape=(72, 72)
        )

    # ### semantic tree visualization with depth
    # visualizer.visualize_all_depth(tree, tree_depth)
    # visualizer.visualize_subtrees_individually(tree, img_shape=(72, 72))
    # ### raw image crop ###
    # visualizer.visualize_hierarchy()
    # visualizer.visualize_all_levels()
    # # sorting result
    # print_nodes_by_priority(sorted_list, top_k=15)
    #
    # # tree structure
    # print_tree_structure(tree)
    #

    # ### single inference ####
    # sam_output_single = sam3.inference(image, text_target[0])
    # masks = sam_output_single["masks"].cpu()
    # boxes = sam_output_single["boxes"].cpu().numpy()
    # scores = sam_output_single["scores"].cpu().numpy()
    # print(f"Predict bbox: {boxes}")
    # print(f"Predict score: {scores}")
    #
    # #  masks shape
    # if masks.ndim == 4:
    #     masks = masks.squeeze(1)  # [N, H, W]
    # else:
    #     masks = masks[None, :, :] if masks.ndim == 2 else masks
    #
    #
    # image_features = sam_output_single['backbone_out']['vision_features']
    # if isinstance(image_features, torch.Tensor):
    #     feat = image_features.detach().cpu().numpy()
    # else:
    #     feat = image_features
    # feat = feat.squeeze(0)  # batch -> (256, 72, 72), C,H,W
    #
    # tree_depth = 2
    # t0 = time.time()
    # builder = ConstrainedTreeBuilder(feat, n_atoms=600, pos_weight=3.5)
    # tree = builder.build_tree(max_depth=tree_depth, min_splits=4, max_splits=8)
    # t1 = time.time()
    #
    # feat_shape = feat.shape
    # visualizer = TreeVisualizer(tree, image, feat_shape)
    #
    # ### mask
    # visualizer.visualize_mask(image, text_target[0], masks)
    # ### feature map PCA-3
    # visualizer.visualize_pca(feat)
    # ### semantic tree visualization with depth
    # visualizer.visualize_all_depth(tree, tree_depth)
    # visualizer.visualize_subtrees_individually(tree, img_shape=(72, 72))
    # ### raw image crop ###
    # visualizer.visualize_hierarchy()






