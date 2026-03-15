use anyhow::Result;
use ndarray::Array1;
use rusqlite::{params, Connection};
use std::path::Path;

pub struct FaceDb {
    conn: Connection,
}

impl FaceDb {
    pub fn open() -> Result<Self> {
        let conn = Connection::open("face_cache.db")?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS face_cache (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                file_mtime INTEGER,
                embedding BLOB
            )",
            [],
        )?;
        Ok(Self { conn })
    }

    pub fn get_cached_embedding(&self, path: &Path, current_mtime: i64) -> Result<Option<Array1<f32>>> {
        let mut stmt = self.conn.prepare(
            "SELECT file_mtime, embedding FROM face_cache WHERE file_path = ?"
        )?;
        
        let path_str = path.to_string_lossy().to_string();
        let mut rows = stmt.query(params![path_str])?;

        if let Some(row) = rows.next()? {
            let mtime: i64 = row.get(0)?;
            if mtime == current_mtime {
                let blob: Vec<u8> = row.get(1)?;
                let embedding: Vec<f32> = bincode::deserialize(&blob)?;
                return Ok(Some(Array1::from_vec(embedding)));
            }
        }
        Ok(None)
    }

    pub fn save_embedding(&self, path: &Path, mtime: i64, embedding: &Array1<f32>) -> Result<()> {
        let path_str = path.to_string_lossy().to_string();
        let embedding_vec: Vec<f32> = embedding.to_vec();
        let blob = bincode::serialize(&embedding_vec)?;

        self.conn.execute(
            "INSERT OR REPLACE INTO face_cache (file_path, file_mtime, embedding) VALUES (?, ?, ?)",
            params![path_str, mtime, blob],
        )?;
        Ok(())
    }
}
