import uuid
import numpy as np
from datetime import datetime

class VisitorState:
    def __init__(self, visitor_id, embedding, timestamp, max_lost_frames=30):
        self.visitor_id = visitor_id
        self.embedding = embedding
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.lost_frames = 0
        self.max_lost_frames = max_lost_frames
        
        # State machine: NEW -> ACTIVE -> LOST -> EXITED
        self.state = "NEW"
        self.has_logged_entry = False

    def update(self, embedding, timestamp):
        # Update embedding with moving average to adapt to appearance changes
        self.embedding = 0.9 * self.embedding + 0.1 * embedding
        self.embedding /= np.linalg.norm(self.embedding)
        
        self.last_seen = timestamp
        self.lost_frames = 0
        
        if self.state in ["NEW", "LOST"]:
            self.state = "ACTIVE"

    def mark_lost(self):
        self.lost_frames += 1
        if self.lost_frames > self.max_lost_frames:
            self.state = "EXITED"
        else:
            self.state = "LOST"

class FaceTracker:
    def __init__(self, db_manager, system_logger, sim_threshold=0.55, max_lost_frames=30):
        self.db = db_manager
        self.logger = system_logger
        self.sim_threshold = sim_threshold
        self.max_lost_frames = max_lost_frames
        
        # Active visitors currently in tracking memory
        self.active_visitors = {}
        
        # Load existing identities from DB to memory for fast matching
        self.known_faces = self.db.get_all_registered_embeddings()
        for vid, emb in self.known_faces.items():
            self.known_faces[vid] = np.array(emb, dtype=np.float32)

    def process_frame_detections(self, frame_boxes_embeddings, timestamp, frame):
        """
        frame_boxes_embeddings: List of dicts {'box': [x1, y1, x2, y2, conf], 'embedding': array, 'crop': img}
        """
        # Set all active to lost initially; will reset if matched
        for vid in self.active_visitors.values():
            if vid.state != "EXITED":
                vid.mark_lost()

        for det in frame_boxes_embeddings:
            box = det['box']
            emb = det['embedding']
            crop = det['crop']

            matched_id = self._match_embedding(emb)

            if matched_id:
                # Identity known
                if matched_id in self.active_visitors:
                    visitor = self.active_visitors[matched_id]
                    visitor.update(emb, timestamp)
                else:
                    # Known in DB, but returning to the frame after being EXITED
                    visitor = VisitorState(matched_id, emb, timestamp, self.max_lost_frames)
                    visitor.state = "ACTIVE"
                    self.active_visitors[matched_id] = visitor
                    self.db.update_visitor_last_seen(matched_id, timestamp)
            else:
                # Completely new face -> Auto Registration
                matched_id = f"VISITOR_{uuid.uuid4().hex[:8].upper()}"
                self.known_faces[matched_id] = emb
                self.db.register_visitor(matched_id, emb, timestamp)
                
                visitor = VisitorState(matched_id, emb, timestamp, self.max_lost_frames)
                visitor.state = "NEW"
                self.active_visitors[matched_id] = visitor
                self.logger.log_info(f"Auto-registered new visitor: {matched_id}")

            # Fire entry event exactly once per visit session
            visitor = self.active_visitors[matched_id]
            if not visitor.has_logged_entry and visitor.state == "ACTIVE":
                img_path = self.logger.save_face_image(crop, matched_id, "ENTRY", timestamp)
                self.db.log_event(matched_id, "ENTRY", timestamp, img_path)
                self.logger.log_info(f"ENTRY logged for {matched_id}")
                visitor.has_logged_entry = True

        # Process exits
        exited_ids = []
        for vid, visitor in self.active_visitors.items():
            if visitor.state == "EXITED":
                img_path = self.logger.save_face_image(None, vid, "EXIT", timestamp)
                self.db.log_event(vid, "EXIT", timestamp, img_path)
                self.logger.log_info(f"EXIT logged for {vid}")
                self.db.update_visitor_last_seen(vid, timestamp)
                exited_ids.append(vid)

        # Cleanup exited visitors from active memory
        for vid in exited_ids:
            del self.active_visitors[vid]

    def _match_embedding(self, embedding):
        """ Find best match above threshold in known faces """
        if not self.known_faces:
            return None

        best_match = None
        best_sim = -1.0

        for vid, known_emb in self.known_faces.items():
            sim = np.dot(embedding, known_emb)
            if sim > best_sim:
                best_sim = sim
                best_match = vid

        if best_sim >= self.sim_threshold:
            return best_match
        return None
