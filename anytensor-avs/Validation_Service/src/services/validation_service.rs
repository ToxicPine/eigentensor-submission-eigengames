use crate::services::oracle_service;
use ndarray::{Array, IxDyn};
use crate::services::oracle_service::{TaskInput, OracleResponse};

/// Deserialize tensor data from bytes that were serialized using the Python TensorSerializer.
pub fn tensor_from_bytes(data: &[u8]) -> Result<Array<f64, IxDyn>, String> {
    // Split data by newlines to extract metadata
    let mut parts = data.splitn(3, |&b| b == b'\n');
    
    // Parse shape
    let shape_bytes = parts.next().ok_or_else(|| "Missing shape information".to_string())?;
    let shape_str = std::str::from_utf8(shape_bytes)
        .map_err(|e| format!("Invalid shape string: {}", e))?;
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to parse shape: {}", e))?;
    
    // Parse dtype
    let dtype_bytes = parts.next().ok_or_else(|| "Missing dtype information".to_string())?;
    let dtype_str = std::str::from_utf8(dtype_bytes)
        .map_err(|e| format!("Invalid dtype string: {}", e))?;
    
    // Get raw data
    let raw_data = parts.next().ok_or_else(|| "Missing tensor data".to_string())?;
    
    match dtype_str {
        "float64" => {
            let mut values = Vec::new();
            let mut chunks = raw_data.chunks_exact(8); // f64 is 8 bytes
            for chunk in &mut chunks {
                if chunk.len() == 8 {
                    let bytes: [u8; 8] = chunk.try_into()
                        .map_err(|_| "Failed to convert chunk to bytes".to_string())?;
                    values.push(f64::from_le_bytes(bytes));
                }
            }
            
            // Create ndarray with the parsed shape
            let array = Array::from_shape_vec(
                IxDyn(&shape), 
                values
            ).map_err(|e| format!("Failed to create ndarray: {}", e))?;
            
            Ok(array)
        },
        // Add support for other data types as needed
        _ => Err(format!("Unsupported dtype: {}", dtype_str))
    }
}

/// Computes the Manhattan (L1) distance between two n-dimensional arrays.
/// 
/// Returns the sum of absolute differences between corresponding elements.
/// Arrays must have the same shape.
pub fn manhattan_distance(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>) -> Result<f64, String> {
    // Check if shapes match
    if a.shape() != b.shape() {
        return Err(format!(
            "Arrays have different shapes: {:?} and {:?}",
            a.shape(), b.shape()
        ));
    }

    // Calculate Manhattan distance
    let distance = a
        .iter()
        .zip(b.iter())
        .map(|(&a_val, &b_val)| (a_val - b_val).abs())
        .sum();

    Ok(distance)
}

pub async fn validate(task_inputs: &TaskInput, proof_of_task: &[u8]) -> Result<bool, String> {
    // Convert the proofOfTask string into a float
    let task_result = match tensor_from_bytes(proof_of_task) {
        Ok(tensor) => tensor,
        Err(_) => return Err("Failed To Parse Proof of Task Tensor".to_string()),
    };
    // Request tensor computation from oracle service
    let oracle_response = oracle_service::compute_tensor(&task_inputs).await;

    match oracle_response {
        // Oracle successfully computed the tensor
        Ok(OracleResponse::Success { data }) => {
            // Convert oracle's integer result to tensor format
            let oracle_tensor = match tensor_from_bytes(&data.to_le_bytes()) {
                Ok(tensor) => tensor,
                Err(e) => return Err(format!(
                    "Failed to convert oracle result to tensor format: {}", e
                )),
            };
            
            // Calculate distance between submitted proof and oracle's computation
            let distance = manhattan_distance(&task_result, &oracle_tensor)
                .map_err(|e| format!("Error calculating distance between tensors: {}", e))?;
            
            // Define maximum acceptable difference between tensors
            let max_allowed_difference = 1e-3;
            
            // Return true if tensors are close enough, false otherwise
            Ok(distance < max_allowed_difference)
        },
        // Oracle encountered an error during computation
        Ok(OracleResponse::Error { message }) => {
            Err(format!("Oracle service failed to compute tensor: {}", message))
        },
        // Network/connection error occurred
        Err(e) => Err(format!("Failed to communicate with oracle service: {}", e)),
    }
}