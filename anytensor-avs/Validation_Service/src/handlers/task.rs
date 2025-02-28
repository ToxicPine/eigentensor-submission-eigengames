use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use log::{info, error};
use crate::services::validation_service;
use crate::services::oracle_service::TaskInput;

#[derive(Deserialize)]
pub struct ValidateRequest {
    pub proof_of_task: Vec<u8>,
    pub task_inputs: TaskInput,
}

#[derive(Serialize)]
pub struct CustomResponse {
    pub data: serde_json::Value,
    pub message: String,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub data: serde_json::Value,
    pub error: bool,
    pub message: String,
}

impl CustomResponse {
    pub fn new(data: serde_json::Value, message: &str) -> Self {
        CustomResponse {
            data,
            message: message.to_string(),
        }
    }
}

impl ErrorResponse {
    pub fn new(data: serde_json::Value, message: &str) -> Self {
        ErrorResponse {
            data,
            error: true,
            message: message.to_string(),
        }
    }
}

// Handler for the `validate` endpoint
pub async fn validate_task(request: web::Json<ValidateRequest>) -> impl Responder {
    let proof_of_task = &request.proof_of_task;
    let task_inputs = &request.task_inputs;
    info!("proofOfTask: {:?}", proof_of_task);

    match validation_service::validate(task_inputs, proof_of_task).await {
        Ok(result) => {
            info!("Vote: {}", if result { "Approve" } else { "Not Approved" });

            let response = CustomResponse::new(
                json!({ "result": result }),
                "Task Validation Successful",
            );

            HttpResponse::Ok().json(response)
        }
        Err(err) => {
            error!("Validation error: {}", err);
            
            let response = ErrorResponse::new(
                json!({}),
                "Error During Task Validation",
            );
            
            HttpResponse::InternalServerError().json(response)
        }
    }
}
