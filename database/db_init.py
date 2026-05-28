"""
数据库初始化脚本 - 从JSON文件构建SQLite穴位数据库
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path


def init_database(db_path: str = "database/acupoint.db",
                  json_sources: list = None):
    """
    初始化SQLite数据库

    Args:
        db_path: 数据库文件路径
        json_sources: JSON数据源列表
    """
    if json_sources is None:
        json_sources = [
            "database/acupoints_torso.json",
            "database/acupoints_limbs.json",
            "database/acupoints_face.json",
            "database/acupoints_hands.json",
        ]

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- 建表 ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS acupoints (
        id TEXT PRIMARY KEY,
        name_cn TEXT NOT NULL,
        name_pinyin TEXT,
        meridian TEXT,
        meridian_code TEXT,
        region TEXT,
        method TEXT,
        data_json TEXT,
        grade TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS population_coefficients (
        acupoint_id TEXT,
        group_key TEXT,
        ratio_modifier REAL DEFAULT 1.0,
        offset_modifier REAL DEFAULT 1.0,
        sample_count INTEGER DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (acupoint_id, group_key)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        acupoint_id TEXT NOT NULL,
        source_image TEXT,
        subject_age INTEGER,
        subject_gender TEXT,
        subject_bmi REAL,
        predicted_x REAL, predicted_y REAL, predicted_z REAL,
        annotated_x REAL, annotated_y REAL, annotated_z REAL,
        delta_x REAL, delta_y REAL, delta_z REAL,
        predicted_ratio REAL,
        annotated_ratio REAL,
        annotator TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS coefficient_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        acupoint_id TEXT,
        group_key TEXT,
        old_ratio REAL,
        new_ratio REAL,
        sample_count INTEGER,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_type TEXT,  -- 'image', 'camera', 'realsense'
        source TEXT,
        total_acupoints_found INTEGER,
        grade_A_count INTEGER DEFAULT 0,
        grade_B_count INTEGER DEFAULT 0,
        grade_C_count INTEGER DEFAULT 0,
        grade_D_count INTEGER DEFAULT 0,
        processing_time_ms REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # --- 导入JSON数据 ---
    total_imported = 0
    for json_path in json_sources:
        if not os.path.exists(json_path):
            print(f"[DB] 跳过不存在的文件: {json_path}")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        region = data.get("region", "unknown")
        count = 0

        for ap in data.get("acupoints", []):
            ap_id = ap["id"]
            cursor.execute("""
                INSERT OR REPLACE INTO acupoints
                (id, name_cn, name_pinyin, meridian, meridian_code, region, method, data_json, grade)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ap_id,
                ap.get("name_cn", ""),
                ap.get("name_pinyin", ""),
                ap.get("meridian", ""),
                ap.get("meridian_code", ""),
                region,
                ap.get("location_rule", {}).get("method", ""),
                json.dumps(ap, ensure_ascii=False),
                ap.get("validation", {}).get("grade", "B"),
            ))
            count += 1

        print(f"[DB] 导入 {json_path}: {count} 条 ({region})")
        total_imported += count

    conn.commit()

    # 验证
    cursor.execute("SELECT COUNT(*) FROM acupoints")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT region, COUNT(*) FROM acupoints GROUP BY region")
    regions = cursor.fetchall()

    print(f"\n{'='*50}")
    print(f"[DB] 数据库初始化完成: {db_path}")
    print(f"[DB] 穴位总数: {total}")
    for region, cnt in regions:
        grade_counts = {}
        cursor.execute(
            "SELECT grade, COUNT(*) FROM acupoints WHERE region=? GROUP BY grade",
            (region,)
        )
        for grade, gc in cursor.fetchall():
            grade_counts[grade] = gc
        grade_str = " ".join([f"{g}:{c}" for g, c in sorted(grade_counts.items())])
        print(f"  {region}: {cnt} 穴 ({grade_str})")

    conn.close()
    return total


def query_database(db_path: str = "database/acupoint.db"):
    """查询数据库统计"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n=== 穴位数据库统计 ===")
    cursor.execute("SELECT meridian_code, COUNT(*) as cnt FROM acupoints GROUP BY meridian_code ORDER BY cnt DESC")
    for mc, cnt in cursor.fetchall():
        cursor.execute("SELECT name_cn FROM acupoints WHERE meridian_code=? LIMIT 3", (mc,))
        examples = [r[0] for r in cursor.fetchall()]
        print(f"  {mc}: {cnt}穴 (例: {', '.join(examples)})")

    cursor.execute("SELECT grade, COUNT(*) FROM acupoints GROUP BY grade ORDER BY grade")
    print("\n精度分布:")
    for grade, cnt in cursor.fetchall():
        print(f"  {grade}级: {cnt}穴")

    conn.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    init_database()
    query_database()
