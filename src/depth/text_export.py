from pathlib import Path

def write_colmap_text(recon, output_dir: str):
    """
    Exports a pycolmap.Reconstruction to COLMAP text format.
    This guarantees compatibility with older legacy COLMAP text consumers
    which crash on PyCOLMAP 4.0+ binary formats due to structural changes
    like pose_prior and new model schemas.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. cameras.txt
    with open(out / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(recon.cameras)}\n")
        for cam_id, cam in recon.cameras.items():
            model_name = getattr(cam, "model_name", getattr(getattr(cam, "model", None), "name", "PINHOLE"))
            params = " ".join(map(str, cam.params))
            f.write(f"{cam_id} {model_name} {cam.width} {cam.height} {params}\n")
            
    # 2. images.txt
    with open(out / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(recon.images)}\n")
        
        for img_id, img in recon.images.items():
            # Handle PyCOLMAP 3.x vs 4.x differences
            if hasattr(img, "cam_from_world"):
                # PyCOLMAP 4.x
                cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                qx, qy, qz, qw = cfw.rotation.quat
                qvec = (qw, qx, qy, qz)
                tvec = cfw.translation
            else:
                # PyCOLMAP 3.x
                qvec = img.qvec
                tvec = img.tvec
                
            q_w, q_x, q_y, q_z = qvec
            t_x, t_y, t_z = tvec
            
            f.write(f"{img_id} {q_w} {q_x} {q_y} {q_z} {t_x} {t_y} {t_z} {img.camera_id} {img.name}\n")
            
            # Points2D
            p2d_strs = []
            for p2d in img.points2D:
                x, y = p2d.xy
                p3d_id = p2d.point3D_id if hasattr(p2d, "point3D_id") else getattr(p2d, "point3d_id", -1)
                # COLMAP text format uses -1 for unobserved points, but some bindings use sys.maxsize
                if p3d_id == 18446744073709551615 or p3d_id < 0:
                    p3d_id = -1
                p2d_strs.append(f"{x} {y} {p3d_id}")
            f.write(" ".join(p2d_strs) + "\n")
            
    # 3. points3D.txt
    with open(out / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(recon.points3D)}\n")
        
        for p3d_id, p3d in recon.points3D.items():
            x, y, z = p3d.xyz
            r, g, b = p3d.color
            err = p3d.error
            
            # Track
            track_strs = []
            if hasattr(p3d, "track"):
                elements = getattr(p3d.track, "elements", p3d.track)
                for el in elements:
                    track_strs.append(f"{el.image_id} {el.point2D_idx}")
                    
            f.write(f"{p3d_id} {x} {y} {z} {r} {g} {b} {err} ")
            f.write(" ".join(track_strs) + "\n")

