use reqwest::Error;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

#[derive(Deserialize, Serialize, Debug)]
pub struct TaskInput {
    pub task_uuid: Uuid,
    pub weights_uuid: Uuid,
    pub input_tensors: HashMap<String, Vec<u8>>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "status")]
pub enum OracleResponse {
    #[serde(rename = "success")]
    Success { data: i32 },
    #[serde(rename = "error")] 
    Error { message: String }
}

pub async fn compute_tensor(task_input: &TaskInput) -> Result<OracleResponse, Error> {
    let client = reqwest::Client::new();
    let port = std::env::var("ANYTENSOR_PORT").unwrap_or_else(|_| "4444".to_string());
    let url = format!("http://localhost:{}/execute", port);
    let response = client.post(url)
        .json(&task_input)
        .send()
        .await?;

    Ok(response.json::<OracleResponse>().await?)
}
