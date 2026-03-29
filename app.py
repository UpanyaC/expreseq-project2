import os
import json
import base64
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.distance import cosine
from deepface import DeepFace

from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash



db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ProtectedApp(db.Model):
    __tablename__ = "protected_apps"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    app_name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ExpressionProfile(db.Model):
    __tablename__ = "expression_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    app_id = db.Column(db.Integer, nullable=False)
    video_path = db.Column(db.String(255), nullable=True)
    face_embedding_path = db.Column(db.Text, nullable=True)
    expression_metadata = db.Column(db.Text, nullable=True)
    landmark_metadata = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AccessAttempt(db.Model):
    __tablename__ = "access_attempts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    app_id = db.Column(db.Integer, nullable=False)
    ip_address = db.Column(db.String(100), nullable=True)
    device_info = db.Column(db.String(255), nullable=True)
    location_info = db.Column(db.String(255), nullable=True)

    status = db.Column(db.String(50), default="pending")
    score = db.Column(db.Float, nullable=True)

    face_score_raw = db.Column(db.Float, nullable=True)
    expression_score_raw = db.Column(db.Float, nullable=True)
    liveness_score_raw = db.Column(db.Float, nullable=True)
    face_distance_raw = db.Column(db.Float, nullable=True)
    expression_distance_raw = db.Column(db.Float, nullable=True)

    decision_reason = db.Column(db.String(255), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Alert(db.Model):
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    app_id = db.Column(db.Integer, nullable=False)
    message = db.Column(db.Text, nullable=False)
    alert_type = db.Column(db.String(50), default="system")
    sent_to = db.Column(db.String(120), nullable=True)
    ip_address = db.Column(db.String(100), nullable=True)
    device_info = db.Column(db.String(255), nullable=True)
    failure_reason = db.Column(db.String(255), nullable=True)
    evidence_image_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

MAX_CAPTURED_FRAMES = 8
MAX_ANALYSIS_FRAMES = 4
MAX_EMOTION_FRAMES = 2
RESIZE_WIDTH = 360
JPEG_QUALITY = 85


def save_frame(frame_b64, path):
    try:
        if not frame_b64 or "," not in frame_b64:
            return False

        _, encoded = frame_b64.split(",", 1)
        img_bytes = base64.b64decode(encoded)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return False

        h, w = img.shape[:2]
        if w > RESIZE_WIDTH:
            new_height = int((RESIZE_WIDTH / w) * h)
            img = cv2.resize(img, (RESIZE_WIDTH, new_height))

        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return True
    except Exception as e:
        print("FRAME SAVE ERROR:", e)
        return False


def select_analysis_frames(frame_paths, max_frames):
    if len(frame_paths) <= max_frames:
        return frame_paths

    indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
    return [frame_paths[i] for i in indices]


def get_embedding(image_path):
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=False
        )
        if result and len(result) > 0:
            return result[0]["embedding"]
    except Exception as e:
        print("EMBEDDING ERROR:", e)

    return None


def get_average_embedding_from_frames(frame_paths):
    embeddings = []

    for path in frame_paths:
        embedding = get_embedding(path)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        return None

    return np.mean(np.array(embeddings), axis=0).tolist()


def compare_embeddings(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 1.0
    return float(cosine(embedding1, embedding2))


def normalize_emotion_vector(emotions):
    vector = np.array([float(emotions.get(k, 0.0)) for k in EMOTION_KEYS], dtype=np.float32)
    total = np.sum(vector)
    if total <= 0:
        return None
    return (vector / total).tolist()


def get_emotion_signature_from_frames(frame_paths):
    emotion_vectors = []
    dominant_labels = []

    for path in frame_paths:
        try:
            result = DeepFace.analyze(
                img_path=path,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv"
            )
            if isinstance(result, list):
                result = result[0]

            emotions = result.get("emotion", {}) if result else {}
            vector = normalize_emotion_vector(emotions)
            if vector is not None:
                emotion_vectors.append(vector)
                dominant_labels.append(result.get("dominant_emotion", "unknown"))
        except Exception as e:
            print("EMOTION ANALYSIS ERROR:", e)

    if not emotion_vectors:
        return None

    avg_vector = np.mean(np.array(emotion_vectors), axis=0)
    dominant = max(set(dominant_labels), key=dominant_labels.count) if dominant_labels else "unknown"

    return {
        "avg_emotions": {k: float(v) for k, v in zip(EMOTION_KEYS, avg_vector.tolist())},
        "dominant_emotion": dominant,
        "frames_used": len(emotion_vectors)
    }


def compare_emotion_signatures(saved_meta, live_meta):
    if not saved_meta or not live_meta:
        return 1.0

    saved_vector = normalize_emotion_vector(saved_meta.get("avg_emotions", {}))
    live_vector = normalize_emotion_vector(live_meta.get("avg_emotions", {}))

    if saved_vector is None or live_vector is None:
        return 1.0

    return float(cosine(saved_vector, live_vector))


def calculate_liveness_score(frame_paths):
    if len(frame_paths) < 2:
        return 0.0

    motion_scores = []
    previous_gray = None

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if previous_gray is not None:
            diff = cv2.absdiff(previous_gray, gray)
            motion_scores.append(float(np.mean(diff)) / 255.0)

        previous_gray = gray

    if not motion_scores:
        return 0.0

    return float(np.mean(motion_scores))


def save_first_image_as_evidence(frame_paths, image_output_path):
    try:
        if not frame_paths:
            return False

        source_path = frame_paths[0]
        img = cv2.imread(source_path)
        if img is None:
            return False

        cv2.imwrite(image_output_path, img)
        return True
    except Exception as e:
        print("EVIDENCE IMAGE ERROR:", e)
        return False


def create_app():
    app = Flask(__name__)
    config_name = os.environ.get("APP_CONFIG", "DevelopmentConfig")

    if config_name == "ProductionConfig":
        app.config.from_object("config.ProductionConfig")
    else:
        app.config.from_object("config.DevelopmentConfig")

    db.init_app(app)

    with app.app_context():
        db.create_all()

        try:
            warmup_img = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(img_path=warmup_img, model_name="Facenet", enforce_detection=False)
        except Exception as e:
            print("DEEPFACE WARMUP WARNING:", e)

    def current_user():
        uid = session.get("user_id")
        if not uid:
            return None
        return db.session.get(User, uid)

    @app.route("/")
    def home():
        user = current_user()
        if session.get("user_id") and user:
            return redirect(url_for("dashboard"))

        if session.get("user_id") and not user:
            session.clear()

        return render_template("index.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "").strip()

            if not name or not email or not password:
                flash("All fields are required.", "danger")
                return redirect(url_for("register"))

            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash("Email already registered. Please login.", "warning")
                return redirect(url_for("login"))

            new_user = User(
                name=name,
                email=email,
                password_hash=generate_password_hash(password)
            )
            db.session.add(new_user)
            db.session.commit()

            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "").strip()

            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password_hash, password):
                session["user_id"] = user.id
                session["user_name"] = user.name
                flash("Login successful.", "success")
                return redirect(url_for("dashboard"))

            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("home"))

    @app.route("/dashboard")
    def dashboard():
        user = current_user()
        if not user:
           session.clear()
           return redirect(url_for("home"))

        apps = ProtectedApp.query.filter_by(user_id=user.id).all()

        app_data = []
        for app_item in apps:
            last_attempt = (
                AccessAttempt.query
                .filter_by(user_id=user.id, app_id=app_item.id)
                .order_by(AccessAttempt.created_at.desc())
                .first()
            )

            app_data.append({
                "app": app_item,
                "last_attempt": last_attempt
            })

        return render_template("dashboard.html", user=user, app_data=app_data)

    @app.route("/alerts")
    def alerts():
        user = current_user()
        if not user:
            session.clear()
            return redirect(url_for("home"))

        alerts = Alert.query.filter_by(user_id=user.id).order_by(Alert.created_at.desc()).all()
        return render_template("alerts.html", alerts=alerts)

    @app.route("/add_app", methods=["GET", "POST"])
    def add_app():
        user = current_user()
        if not user:
            session.clear()
            return redirect(url_for("home"))

        if request.method == "POST":
            app_name = request.form.get("app_name", "").strip()

            if not app_name:
                flash("App name is required.", "danger")
                return redirect(url_for("add_app"))

            new_app = ProtectedApp(user_id=user.id, app_name=app_name)
            db.session.add(new_app)
            db.session.commit()

            flash(f"{app_name} added successfully!", "success")
            return redirect(url_for("dashboard"))

        return render_template("add_app.html")

    @app.route("/record_expression/<int:app_id>", methods=["GET", "POST"])
    def record_expression(app_id):
        user = current_user()
        if not user:
            session.clear()
            return redirect(url_for("home"))

        app_obj = ProtectedApp.query.filter_by(id=app_id, user_id=user.id).first()
        if not app_obj:
            flash("Protected app not found.", "danger")
            return redirect(url_for("dashboard"))

        existing_profile = ExpressionProfile.query.filter_by(
            user_id=user.id,
            app_id=app_id
        ).first()

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            frames = data.get("frames", [])

            if not isinstance(frames, list) or len(frames) < 4:
                flash("Please capture at least 4 clear frames.", "danger")
                return redirect(url_for("record_expression", app_id=app_id))

            folder = os.path.join(
                app.config["UPLOAD_FOLDER"],
                "enrollments",
                f"user_{user.id}",
                f"app_{app_id}"
            )
            os.makedirs(folder, exist_ok=True)

            saved_frame_paths = []
            for i, frame_b64 in enumerate(frames[:MAX_CAPTURED_FRAMES], start=1):
                frame_path = os.path.join(folder, f"frame_{i:03d}.jpg")
                if save_frame(frame_b64, frame_path):
                    saved_frame_paths.append(frame_path)

            if len(saved_frame_paths) < 4:
                flash("Enrollment frames could not be saved clearly. Please try again.", "danger")
                return redirect(url_for("record_expression", app_id=app_id))

            embedding_frame_paths = select_analysis_frames(saved_frame_paths, MAX_ANALYSIS_FRAMES)
            emotion_frame_paths = select_analysis_frames(saved_frame_paths, MAX_EMOTION_FRAMES)

            try:
                embedding = get_average_embedding_from_frames(embedding_frame_paths)
                emotion_signature = get_emotion_signature_from_frames(emotion_frame_paths)
                liveness_score = calculate_liveness_score(embedding_frame_paths)
            except Exception as e:
                print("ENROLLMENT PIPELINE ERROR:", e)
                flash("Enrollment processing failed. Please try again.", "warning")
                return redirect(url_for("record_expression", app_id=app_id))

            if embedding is None:
                flash("Enrollment frames could not be processed clearly. Please try again in better lighting.", "danger")
                return redirect(url_for("record_expression", app_id=app_id))

            relative_reference_path = os.path.join(
                "static", "uploads", "videos", "enrollments", f"user_{user.id}", f"app_{app_id}", "frame_001.jpg"
            ).replace("\\", "/")

            expression_metadata = {
                "frames_saved": len(saved_frame_paths),
                "emotion_signature": emotion_signature,
                "enrollment_liveness": liveness_score,
            }

            landmark_metadata = {
                "liveness_mode": "motion_based_v1"
            }

            if existing_profile:
                existing_profile.video_path = relative_reference_path
                existing_profile.face_embedding_path = json.dumps(embedding)
                existing_profile.expression_metadata = json.dumps(expression_metadata)
                existing_profile.landmark_metadata = json.dumps(landmark_metadata)
            else:
                existing_profile = ExpressionProfile(
                    user_id=user.id,
                    app_id=app_id,
                    video_path=relative_reference_path,
                    face_embedding_path=json.dumps(embedding),
                    expression_metadata=json.dumps(expression_metadata),
                    landmark_metadata=json.dumps(landmark_metadata)
                )
                db.session.add(existing_profile)

            db.session.commit()
            flash(f"Expression profile saved successfully for {app_obj.app_name}.", "success")
            return redirect(url_for("dashboard"))

        return render_template("record_expression.html", app=app_obj, existing_profile=existing_profile)

    @app.route("/simulate_access/<int:app_id>")
    def simulate_access(app_id):
        user = current_user()
        if not user:
            session.clear()
            return redirect(url_for("home"))

        app_obj = ProtectedApp.query.filter_by(id=app_id, user_id=user.id).first()
        if not app_obj:
            flash("Protected app not found.", "danger")
            return redirect(url_for("dashboard"))

        attempt = AccessAttempt(
            user_id=user.id,
            app_id=app_id,
            ip_address=request.remote_addr,
            device_info=request.user_agent.string,
            location_info="Unknown",
            status="challenge_required"
        )
        db.session.add(attempt)
        db.session.commit()

        flash(f"Suspicious access attempt created for {app_obj.app_name}.", "warning")
        return redirect(url_for("verify_access", app_id=app_id, attempt_id=attempt.id))
     
    @app.route("/open_app/<int:app_id>")
    def open_app(app_id):
        user = current_user()
        if not user:
            session.clear()
            return redirect(url_for("home"))

        app_obj = ProtectedApp.query.filter_by(id=app_id, user_id=user.id).first()
        if not app_obj:
            flash("Protected app not found.", "danger")
            return redirect(url_for("dashboard"))

        saved_profile = ExpressionProfile.query.filter_by(
            user_id=user.id,
            app_id=app_id
        ).first()

        if not saved_profile:
            flash("Please record your expression for this app before opening it.", "warning")
            return redirect(url_for("record_expression", app_id=app_id))

        attempt = AccessAttempt(
            user_id=user.id,
            app_id=app_id,
            ip_address=request.remote_addr,
            device_info=request.user_agent.string,
            location_info="Unknown",
            status="challenge_required"
        )
        db.session.add(attempt)
        db.session.commit()

        return redirect(url_for("verify_access", app_id=app_id, attempt_id=attempt.id))

    @app.route("/protected_app/<int:app_id>")
    def protected_app(app_id):
        user = current_user()
        if not user:
            session.clear()
            return redirect(url_for("home"))

        if session.get("verified_app") != app_id:
            flash("Unauthorized access. Please verify first.", "danger")
            return redirect(url_for("dashboard"))

        app_obj = ProtectedApp.query.filter_by(id=app_id, user_id=user.id).first()
        if not app_obj:
            flash("Protected app not found.", "danger")
            return redirect(url_for("dashboard"))

        app_name = app_obj.app_name.strip().lower()

        template_map = {
            "whatsapp": "apps/whatsapp.html",
            "linkedin": "apps/linkedin.html",
            "instagram": "apps/instagram.html",
            "youtube": "apps/youtube.html",
            "bank": "apps/bank.html",
        }

        template_name = template_map.get(app_name, "apps/generic_app.html")
        return render_template(template_name, app=app_obj, user=user)

    @app.route("/verify_access/<int:app_id>/<int:attempt_id>", methods=["GET", "POST"])
    def verify_access(app_id, attempt_id):
        user = current_user()
        if not user:
            session.clear()
            return redirect(url_for("home"))

        app_obj = ProtectedApp.query.filter_by(id=app_id, user_id=user.id).first()
        attempt = AccessAttempt.query.filter_by(id=attempt_id, user_id=user.id, app_id=app_id).first()

        if not app_obj or not attempt:
            flash("Verification session not found.", "danger")
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            frames = data.get("frames", [])

            if not isinstance(frames, list) or len(frames) < 4:
                flash("Please capture at least 4 verification frames.", "danger")
                return redirect(url_for("verify_access", app_id=app_id, attempt_id=attempt_id))

            verification_folder = os.path.join(
                app.config["UPLOAD_FOLDER"],
                "live_checks",
                f"user_{user.id}",
                f"attempt_{attempt_id}"
            )
            os.makedirs(verification_folder, exist_ok=True)

            saved_frame_paths = []
            for i, frame_b64 in enumerate(frames[:MAX_CAPTURED_FRAMES], start=1):
                frame_path = os.path.join(verification_folder, f"live_{i:03d}.jpg")
                if save_frame(frame_b64, frame_path):
                    saved_frame_paths.append(frame_path)

            saved_profile = ExpressionProfile.query.filter_by(
                user_id=user.id,
                app_id=app_id
            ).first()

            if not saved_profile or not saved_profile.face_embedding_path:
                flash("No enrolled profile found for this app. Please record expression first.", "danger")
                return redirect(url_for("record_expression", app_id=app_id))

            enrolled_embedding = json.loads(saved_profile.face_embedding_path)
            saved_expression_meta = json.loads(saved_profile.expression_metadata or "{}")

            embedding_frame_paths = select_analysis_frames(saved_frame_paths, MAX_ANALYSIS_FRAMES)
            emotion_frame_paths = select_analysis_frames(saved_frame_paths, MAX_EMOTION_FRAMES)

            try:
                live_embedding = get_average_embedding_from_frames(embedding_frame_paths)
                live_emotion_signature = get_emotion_signature_from_frames(emotion_frame_paths)
                live_liveness_score = calculate_liveness_score(embedding_frame_paths)
            except Exception as e:
                print("VERIFICATION PIPELINE ERROR:", e)
                attempt.status = "retry"
                attempt.decision_reason = "verification pipeline error"
                db.session.commit()
                flash("Verification could not be completed. Please try again.", "warning")
                return redirect(url_for("verify_access", app_id=app_id, attempt_id=attempt_id))

            if live_embedding is None:
                attempt.status = "retry"
                attempt.decision_reason = "live embedding not generated"
                db.session.commit()
                flash("Verification frames could not be processed clearly. Please try again.", "warning")
                return redirect(url_for("verify_access", app_id=app_id, attempt_id=attempt_id))

            face_distance = compare_embeddings(enrolled_embedding, live_embedding)
            expression_distance = compare_emotion_signatures(
                saved_expression_meta.get("emotion_signature"),
                live_emotion_signature
            )

            face_score = max(0.0, 1.0 - face_distance)
            expression_score = max(0.0, 1.0 - expression_distance)
            liveness_score = min(1.0, live_liveness_score * 12.0)

            combined_score = (face_score * 0.55) + (expression_score * 0.25) + (liveness_score * 0.20)

            attempt.score = combined_score
            attempt.face_score_raw = face_score
            attempt.expression_score_raw = expression_score
            attempt.liveness_score_raw = liveness_score
            attempt.face_distance_raw = face_distance
            attempt.expression_distance_raw = expression_distance

            print("face_distance:", face_distance)
            print("expression_distance:", expression_distance)
            print("live_liveness_score:", live_liveness_score)
            print("face_score:", face_score)
            print("expression_score:", expression_score)
            print("liveness_score:", liveness_score)
            print("combined_score:", combined_score)

            if live_liveness_score < 0.010:
                failure_reason = "low liveness detected"
                decision = "retry"
            elif face_distance <= 0.50 and combined_score >= 0.68:
                failure_reason = None
                decision = "verified"
            elif face_distance <= 0.65 and combined_score >= 0.52:
                failure_reason = "verification inconclusive"
                decision = "retry"
            else:
                failure_reason = "face or expression mismatch"
                decision = "denied"

            print("decision:", decision)
            print("reason:", failure_reason)

            if decision == "verified":
                attempt.status = "verified"
                attempt.decision_reason = "verified"
                db.session.commit()

                session["verified_app"] = app_id

                flash(
                     f"Access granted | face={face_score:.2f}, expression={expression_score:.2f}, liveness={liveness_score:.2f}",
                     "success"
                )
                return redirect(url_for("protected_app", app_id=app_id))

            if decision == "retry":
                attempt.status = "retry"
                attempt.decision_reason = failure_reason or "verification inconclusive"
                db.session.commit()
                flash(
                    f"Verification inconclusive. Please try again | face={face_score:.2f}, expression={expression_score:.2f}, liveness={liveness_score:.2f}",
                    "warning"
                )
                return redirect(url_for("verify_access", app_id=app_id, attempt_id=attempt_id))

            attempt.status = "denied"
            attempt.decision_reason = failure_reason or "verification denied"
            db.session.commit()

            evidence_image_folder = os.path.join(app.config["UPLOAD_FOLDER"], "alert_images")
            os.makedirs(evidence_image_folder, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            image_filename = f"alert_user_{user.id}_app_{app_id}_{attempt_id}_{timestamp}.jpg"
            image_save_path = os.path.join(evidence_image_folder, image_filename)
            image_saved = save_first_image_as_evidence(saved_frame_paths, image_save_path)

            relative_image_path = None
            if image_saved:
                relative_image_path = os.path.join(
                    "static", "uploads", "videos", "alert_images", image_filename
                ).replace("\\", "/")

            alert = Alert(
                user_id=user.id,
                app_id=app_id,
                message=f"ALERT: {app_obj.app_name} blocked. Unauthorized access detected.",
                alert_type="system",
                sent_to=user.email,
                ip_address=request.remote_addr,
                device_info=request.user_agent.string,
                failure_reason=failure_reason,
                evidence_image_path=relative_image_path
            )
            db.session.add(alert)
            db.session.commit()

            flash(
                f"ALERT: {app_obj.app_name} blocked | reason={failure_reason} | face={face_score:.2f}, expression={expression_score:.2f}, liveness={liveness_score:.2f}",
                "danger"
            )
            return redirect(url_for("dashboard"))

        return render_template("verify_access.html", app=app_obj, attempt=attempt)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
