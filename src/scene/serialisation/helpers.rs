use bytemuck::{Pod, Zeroable};
use std::io::{Read, Result as IoResult, Write};

pub fn serialize_pod<T: Pod>(value: &T, writer: &mut dyn Write) -> IoResult<()> {
    writer.write_all(bytemuck::bytes_of(value))
}

pub fn deserialize_pod<T: Pod + Zeroable>(reader: &mut dyn Read) -> IoResult<T> {
    let mut value = T::zeroed();
    reader.read_exact(bytemuck::bytes_of_mut(&mut value))?;
    Ok(value)
}

pub fn serialize_vec_pod<T: Pod>(vec: &[T], writer: &mut dyn Write) -> IoResult<()> {
    writer.write_all(&(vec.len() as u64).to_le_bytes())?;
    writer.write_all(bytemuck::cast_slice(vec))
}

pub fn deserialize_vec_pod<T: Pod + Zeroable>(reader: &mut dyn Read) -> IoResult<Vec<T>> {
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes) as usize;

    let mut vec = vec![T::zeroed(); len];
    reader.read_exact(bytemuck::cast_slice_mut(&mut vec))?;
    Ok(vec)
}

pub fn serialize_string(s: &str, writer: &mut dyn Write) -> IoResult<()> {
    let bytes = s.as_bytes();
    writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
    writer.write_all(bytes)
}

pub fn deserialize_string(reader: &mut dyn Read) -> IoResult<String> {
    let mut len_bytes = [0u8; 8];
    reader.read_exact(&mut len_bytes)?;
    let len = u64::from_le_bytes(len_bytes) as usize;

    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::{Pod, Zeroable};
    use std::io::Cursor;

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct TestPod {
        a: f32,
        b: u32,
    }

    unsafe impl Zeroable for TestPod {}
    unsafe impl Pod for TestPod {}

    #[test]
    fn pod_roundtrip() {
        let value = TestPod { a: 1.25, b: 42 };
        let mut buffer = Vec::new();

        serialize_pod(&value, &mut buffer).unwrap();
        let mut cursor = Cursor::new(buffer);
        let out = deserialize_pod::<TestPod>(&mut cursor).unwrap();

        assert_eq!(value, out);
    }

    #[test]
    fn vec_pod_roundtrip() {
        let vec = vec![
            TestPod { a: 1.0, b: 1 },
            TestPod { a: 2.0, b: 2 },
            TestPod { a: 3.0, b: 3 },
        ];

        let mut buffer = Vec::new();
        serialize_vec_pod(&vec, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let out = deserialize_vec_pod::<TestPod>(&mut cursor).unwrap();

        assert_eq!(vec, out);
    }

    #[test]
    fn string_roundtrip() {
        let s = "hello engine serialization";

        let mut buffer = Vec::new();
        serialize_string(s, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let out = deserialize_string(&mut cursor).unwrap();

        assert_eq!(s, out);
    }

    #[test]
    fn empty_vec_roundtrip() {
        let vec: Vec<u32> = Vec::new();

        let mut buffer = Vec::new();
        serialize_vec_pod(&vec, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let out = deserialize_vec_pod::<u32>(&mut cursor).unwrap();

        assert!(out.is_empty());
    }
}
