use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use crate::services::oracle_service;

#[derive(Debug, Clone)]
pub(crate) struct TaskId(Uuid);

#[derive(Debug, Clone)]
pub(crate) struct WeightsId(Uuid);


#[derive(Deserialize)]
pub struct ExecuteTaskPayload {
    pub task_uuid: String,
    pub weights_uuid: String,
    pub input_tensors: HashMap<String, Vec<u8>>,
}

#[derive(Serialize)]
struct CustomResponse {
    status: String,
    data: HashMap<String, serde_json::Value>,
}

pub async fn execute_task(payload: web::Json<ExecuteTaskPayload>) -> impl Responder {
    println!("Executing Task");
    let task_uuid = match Uuid::parse_str(&payload.task_uuid) {
        Ok(uuid) => uuid,
        Err(err) => {
            eprintln!("Failed to parse task UUID: {}", err);
            return HttpResponse::BadRequest().json("Invalid task UUID format");
        }
    };
    let weights_uuid = match Uuid::parse_str(&payload.weights_uuid) {
        Ok(uuid) => uuid,
        Err(err) => {
            eprintln!("Failed to parse weights UUID: {}", err);
            return HttpResponse::BadRequest().json("Invalid weights UUID format"); 
        }
    };

    let task_input = oracle_service::TaskInput {
        task_uuid: task_uuid,
        weights_uuid: weights_uuid,
        input_tensors: payload.input_tensors.clone(),
    };

    match oracle_service::compute_tensor(&task_input).await {
        Ok(result) => {
            HttpResponse::Ok().json(format!("Task {} executed successfully", result.task_uuid))
        }
        Err(err) => {
            eprintln!("Error computing tensor: {}", err);
            HttpResponse::ServiceUnavailable().json("Network error occurred")
        }
    }
}
