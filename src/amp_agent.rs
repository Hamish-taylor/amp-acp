use agent_client_protocol::{
    Agent, AgentCapabilities, AgentSideConnection, AuthMethod, AuthMethodId, AuthenticateRequest,
    AuthenticateResponse, CancelNotification, Client, ContentBlock, Diff, EmbeddedResourceResource,
    Error, ExtNotification, ExtRequest, ExtResponse, InitializeRequest, InitializeResponse,
    LoadSessionRequest, LoadSessionResponse, McpCapabilities, NewSessionRequest,
    NewSessionResponse, Plan, PlanEntry, PlanEntryPriority, PlanEntryStatus, PromptCapabilities,
    PromptRequest, PromptResponse, SessionId, SessionNotification, SessionUpdate,
    SetSessionModeRequest, SetSessionModeResponse, StopReason, TextContent, ToolCall,
    ToolCallContent, ToolCallId, ToolCallLocation, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields, ToolKind, V1,
};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::binary_heap::PeekMut;
use std::fmt::format;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use std::{env, panic};
use tokio::sync::OnceCell;
use tokio::time::sleep;
use tracing::error;

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AmpConversation {
    messages: Vec<AmpMessage>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AmpMessage {
    pub role: String,
    pub content: Vec<AmpContentBlock>,
    pub state: Option<AmpMessageState>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AmpMessageState {
    #[serde(rename = "type")]
    pub t: Option<String>,
    pub stop_reason: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AmpEditFileToolCall {
    pub path: String,
    pub old_str: Option<String>,
    pub new_str: String,
}
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum AmpContentBlock {
    Text(AmpTextContentBlock),
    Thinking(AmpThinkingContentBlock),
    ToolUse(AmpToolUseContentBlock),
    ToolResult(AmpToolResultContentBlock),
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AmpTextContentBlock {
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AmpThinkingContentBlock {
    pub thinking: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AmpToolUseContentBlock {
    pub id: String,
    pub name: AmpTool,
    pub input: serde_json::Value,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AmpToolResultContentBlock {
    #[serde(rename = "toolUseID")]
    pub tool_use_id: String,
    pub run: serde_json::Value,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub struct AmpPlanWriteToolCall {
    pub todos: Vec<AmpPlanTodo>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub struct AmpPlanTodo {
    pub id: String,
    pub content: String,
    pub status: AmpPlanTodoStatus,
    pub priority: AmpPlanTodoPriority,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum AmpPlanTodoStatus {
    Completed,
    Todo,
    #[serde(rename = "in-progress")]
    InProgress,
}

impl AmpPlanTodoStatus {
    pub fn to_acp_plan_status(&self) -> PlanEntryStatus {
        match self {
            AmpPlanTodoStatus::Completed => PlanEntryStatus::Completed,
            AmpPlanTodoStatus::Todo => PlanEntryStatus::Pending,
            AmpPlanTodoStatus::InProgress => PlanEntryStatus::InProgress,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum AmpPlanTodoPriority {
    High,
    Medium,
    Low,
}

impl AmpPlanTodoPriority {
    pub fn to_acp_plan_priority(&self) -> PlanEntryPriority {
        match self {
            AmpPlanTodoPriority::High => PlanEntryPriority::High,
            AmpPlanTodoPriority::Medium => PlanEntryPriority::Medium,
            AmpPlanTodoPriority::Low => PlanEntryPriority::Low,
        }
    }
}

impl AmpPlanWriteToolCall {
    pub fn to_acp_plan(&self) -> Plan {
        Plan {
            entries: self
                .todos
                .iter()
                .map(|todo| PlanEntry {
                    content: todo.content.clone(),
                    status: todo.status.clone().to_acp_plan_status(),
                    priority: todo.priority.clone().to_acp_plan_priority(),
                    meta: None,
                })
                .collect(),
            meta: None,
        }
    }
}

pub trait AmpDiff<T> {
    fn diff(&self, other: &T) -> Option<T>;
}

impl AmpConversation {
    fn pretty_print(&self) -> String {
        let mut output = String::new();
        for message in &self.messages {
            for i in 0..message.content.len() {
                let block = &message.content[i];
                let mut l = match block {
                    AmpContentBlock::Text(amp_text_content_block) => {
                        format!("{}", amp_text_content_block.text.clone())
                    }
                    AmpContentBlock::Thinking(_) => String::new(),
                    AmpContentBlock::ToolUse(amp_tool_use_content_block) => {
                        format!(
                            "**{}**",
                            &amp_tool_use_content_block
                                .name
                                .to_title(&amp_tool_use_content_block.input)
                        )
                    }
                    AmpContentBlock::ToolResult(_) => String::new(),
                };
                l.push_str(" \n\n");
                output.push_str(&l);
            }
        }

        output
    }
}

impl AmpDiff<AmpConversation> for AmpConversation {
    fn diff(&self, other: &AmpConversation) -> Option<AmpConversation> {
        let num_diff = other.messages.len() - self.messages.len();
        let messages_diff: Vec<Option<AmpMessage>> = self
            .messages
            .iter()
            .zip(other.messages.iter())
            .map(|(a, b)| a.diff(b))
            .collect();

        let mut f: Vec<AmpMessage> = messages_diff.iter().filter_map(|m| m.clone()).collect();

        if num_diff > 0 {
            //take the last num_diff items from other
            let mut rem: Vec<AmpMessage> = other
                .messages
                .iter()
                .cloned()
                .rev()
                .take(num_diff)
                .collect();
            f.append(&mut rem);
        }
        Some(AmpConversation { messages: f })
    }
}

impl AmpDiff<AmpContentBlock> for AmpContentBlock {
    fn diff(&self, other: &AmpContentBlock) -> Option<AmpContentBlock> {
        match (self, other) {
            (AmpContentBlock::Text(a), AmpContentBlock::Text(b)) => {
                if a.text == b.text {
                    None
                } else {
                    Some(AmpContentBlock::Text(AmpTextContentBlock {
                        text: b.text.replace(&a.text, ""),
                    }))
                }
            }
            (AmpContentBlock::Thinking(a), AmpContentBlock::Thinking(b)) => {
                if a.thinking == b.thinking {
                    None
                } else {
                    Some(AmpContentBlock::Thinking(AmpThinkingContentBlock {
                        thinking: b.thinking.replace(&a.thinking, ""),
                    }))
                }
            }
            (AmpContentBlock::ToolUse(a), AmpContentBlock::ToolUse(b)) => {
                if a.id == b.id && a.name == b.name && a.input == b.input {
                    None
                } else {
                    Some(AmpContentBlock::ToolUse(AmpToolUseContentBlock {
                        id: b.id.clone(),
                        name: b.name.clone(),
                        input: b.input.clone(),
                    }))
                }
            }
            (AmpContentBlock::ToolResult(a), AmpContentBlock::ToolResult(b)) => {
                if a.tool_use_id == b.tool_use_id && a.run == b.run {
                    None
                } else {
                    Some(AmpContentBlock::ToolResult(AmpToolResultContentBlock {
                        tool_use_id: b.tool_use_id.clone(),
                        run: b.run.clone(),
                    }))
                }
            }
            _ => None,
        }
    }
}

impl AmpDiff<AmpMessage> for AmpMessage {
    fn diff(&self, other: &AmpMessage) -> Option<AmpMessage> {
        let num_diff = other.content.len() - self.content.len();
        if self.role == other.role {
            let mut content_diff: Vec<AmpContentBlock> = self
                .content
                .iter()
                .zip(other.content.iter())
                .filter_map(|(a, b)| a.diff(b))
                .collect();

            if num_diff > 0 {
                //take the last num_diff items from other
                let mut rem: Vec<AmpContentBlock> =
                    other.content.iter().cloned().rev().take(num_diff).collect();
                content_diff.append(&mut rem);
            }
            Some(AmpMessage {
                role: self.role.clone(),
                content: content_diff,
                state: other.state.clone(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmpTool {
    Bash,
    #[serde(rename = "create_file")]
    CreateFile,
    #[serde(rename = "edit_file")]
    EditFile,
    #[serde(rename = "finder")]
    Finder,
    #[serde(rename = "glob")]
    Glob,
    Grep,
    #[serde(rename = "mermaid")]
    Mermaid,
    #[serde(rename = "oracle")]
    Oracle,
    Read,
    #[serde(rename = "read_mcp_resource")]
    ReadMcpResource,
    #[serde(rename = "read_web_page")]
    ReadWebPage,
    Task,
    #[serde(rename = "todo_read")]
    TodoRead,
    #[serde(rename = "todo_write")]
    TodoWrite,
    #[serde(rename = "undo_edit")]
    UndoEdit,
    #[serde(rename = "web_search")]
    WebSearch,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpReadToolCall {
    path: String,
    read_range: Option<Vec<i32>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpCreateToolCall {
    path: String,
    content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpBashToolCall {
    cmd: String,
    cwd: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpWebSearchToolCall {
    query: String,
    max_results: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpWebReadToolCall {
    url: String,
    prompt: Option<String>,
    raw: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpTaskToolCall {
    prompt: String,
    description: String,
}

impl AmpTool {
    fn to_title(&self, content: &Value) -> String {
        match self {
            AmpTool::Oracle => "Consulting the Oracle".to_string(),
            AmpTool::Read => {
                let tool_call: Result<AmpReadToolCall, serde_json::Error> =
                    serde_json::from_value(content.clone());

                if let Ok(t) = tool_call {
                    let path = PathBuf::from(&t.path);
                    let file_name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or_default();
                    if path.is_file() {
                        format!("Read [{}](file://{})", file_name, t.path)
                    } else {
                        format!("Read {}", t.path)
                    }
                } else {
                    "Reading file".to_string()
                }
            }
            AmpTool::ReadMcpResource => "Read mcp resource".to_string(),
            AmpTool::ReadWebPage => {
                let tool_call: Result<AmpWebReadToolCall, serde_json::Error> =
                    serde_json::from_value(content.clone());

                if let Ok(t) = tool_call {
                    format!("Reading {}", t.url)
                } else {
                    "Read webpage".to_string()
                }
            }
            AmpTool::Task => "Sub agent".to_string(),
            AmpTool::TodoRead => "Todo read".to_string(),
            AmpTool::TodoWrite => "Todo write".to_string(),
            AmpTool::UndoEdit => "Undo edit".to_string(),
            AmpTool::WebSearch => {
                let tool_call: Result<AmpWebSearchToolCall, serde_json::Error> =
                    serde_json::from_value(content.clone());

                if let Ok(t) = tool_call {
                    format!("Searching for \"{}\"", t.query)
                } else {
                    "Web search".to_string()
                }
            }
            AmpTool::Other => "Unknown".to_string(),
            AmpTool::Bash => {
                let tool_call: Result<AmpBashToolCall, serde_json::Error> =
                    serde_json::from_value(content.clone());

                if let Ok(t) = tool_call {
                    t.cmd
                } else {
                    "Bash".to_string()
                }
            }
            AmpTool::CreateFile => {
                let tool_call: Result<AmpCreateToolCall, serde_json::Error> =
                    serde_json::from_value(content.clone());
                if let Ok(t) = tool_call {
                    let path = PathBuf::from(&t.path);
                    let file_name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or_default();

                    format!("Created [{}](file://{})", file_name, t.path)
                } else {
                    "Creating file".to_string()
                }
            }
            AmpTool::EditFile => {
                let data: Result<AmpEditFileToolCall, serde_json::Error> =
                    serde_json::from_value(content.clone());

                if let Ok(data) = data {
                    let path = PathBuf::from(&data.path);
                    let file_name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or_default();
                    format!("Edited [{}](file://{})", file_name, data.path)
                } else {
                    "Editing file".to_string()
                }
            }
            AmpTool::Finder => "Finder".to_string(),
            AmpTool::Glob => "Glob".to_string(),
            AmpTool::Grep => "Grep".to_string(),
            AmpTool::Mermaid => "Mermaid".to_string(),
        }
    }
}

fn amp_tool_to_tool_kind(amp_tool: &AmpTool) -> ToolKind {
    match amp_tool {
        AmpTool::Bash => ToolKind::Execute,
        AmpTool::CreateFile => ToolKind::Edit,
        AmpTool::EditFile => ToolKind::Edit,
        AmpTool::Finder => ToolKind::Search,
        AmpTool::Glob => ToolKind::Execute,
        AmpTool::Grep => ToolKind::Execute,
        AmpTool::Mermaid => ToolKind::Other,
        AmpTool::Oracle => ToolKind::Think,
        AmpTool::Read => ToolKind::Read,
        AmpTool::ReadMcpResource => ToolKind::Fetch,
        AmpTool::ReadWebPage => ToolKind::Fetch,
        AmpTool::Task => ToolKind::Think,
        AmpTool::TodoRead => ToolKind::Think,
        AmpTool::TodoWrite => ToolKind::Think,
        AmpTool::UndoEdit => ToolKind::Edit,
        AmpTool::WebSearch => ToolKind::Search,
        AmpTool::Other => ToolKind::Other,
    }
}

pub struct AmpAgent {
    cwd: Rc<RefCell<Option<PathBuf>>>,
    client: OnceCell<Rc<AgentSideConnection>>,
    amp_command: Rc<RefCell<Option<Child>>>,
    threads_directory: PathBuf,
}

impl AmpAgent {
    pub fn new() -> Self {
        // Todo: Windows support
        let home_dir = env::home_dir().unwrap();
        let threads_directory = format!("{}/.local/share/amp/threads/", home_dir.display());

        Self {
            cwd: Rc::new(RefCell::new(None)),
            client: OnceCell::new(),
            amp_command: Rc::new(RefCell::new(None)),
            threads_directory: PathBuf::from(threads_directory),
        }
    }

    pub fn set_client(&self, client: Rc<AgentSideConnection>) {
        self.client.set(client);
    }

    pub fn set_amp_command(&self, command: Child) {
        self.amp_command.replace(Some(command));
    }

    pub fn client(&self) -> Rc<AgentSideConnection> {
        Rc::clone(self.client.get().expect("Client should be set"))
    }

    pub fn get_amp_thread(&self, thread_id: SessionId) -> Option<AmpConversation> {
        let thread_id_str: &str = &thread_id.0;
        let thread_path = self
            .threads_directory
            .join(format!("{}.json", thread_id_str));

        let mut file = File::open(&thread_path).ok()?;
        let mut contents = String::new();
        file.read_to_string(&mut contents).ok()?;

        serde_json::from_str(&contents).ok()
    }

    async fn process_conversation(
        &self,
        conversation: &AmpConversation,
        session_id: SessionId,
        file_edits: &mut HashMap<String, AmpEditFileToolCall>,
        tasks: &mut HashMap<String, AmpTaskToolCall>,
    ) {
        for message in &conversation.messages {
            for block in &message.content {
                match block {
                    AmpContentBlock::Text(text_content_block) => {
                        if message.role != "user" {
                            let notification = SessionNotification {
                                session_id: session_id.clone(),
                                update: SessionUpdate::AgentMessageChunk {
                                    content: ContentBlock::Text(TextContent {
                                        annotations: None,
                                        text: text_content_block.text.clone(),
                                        meta: None,
                                    }),
                                },
                                meta: None,
                            };

                            if let Err(e) = self.client().session_notification(notification).await {
                                error!("Failed to send session notification: {:?}", e);
                            }
                        }
                    }
                    AmpContentBlock::Thinking(thinking_content_block) => {
                        let notification = SessionNotification {
                            session_id: session_id.clone(),
                            update: SessionUpdate::AgentThoughtChunk {
                                content: ContentBlock::Text(TextContent {
                                    annotations: None,
                                    text: thinking_content_block.thinking.clone(),
                                    meta: None,
                                }),
                            },
                            meta: None,
                        };

                        if let Err(e) = self.client().session_notification(notification).await {
                            error!("Failed to send session notification: {:?}", e);
                        }
                    }
                    AmpContentBlock::ToolUse(tool_use_content_block) => {
                        let title = tool_use_content_block
                            .name
                            .to_title(&tool_use_content_block.input);
                        let mut content = vec![];
                        match tool_use_content_block.name {
                            AmpTool::EditFile => {
                                let data: Result<AmpEditFileToolCall, serde_json::Error> =
                                    serde_json::from_value(tool_use_content_block.input.clone());

                                if let Ok(data) = data {
                                    file_edits
                                        .entry(tool_use_content_block.id.clone())
                                        .or_insert(data);

                                    continue;
                                }
                            }
                            AmpTool::TodoWrite => {
                                let plan: Result<AmpPlanWriteToolCall, serde_json::Error> =
                                    serde_json::from_value(tool_use_content_block.input.clone());

                                if let Ok(plan) = plan {
                                    let notification = SessionNotification {
                                        session_id: session_id.clone(),
                                        update: SessionUpdate::Plan(plan.to_acp_plan()),
                                        meta: None,
                                    };

                                    if let Err(e) =
                                        self.client().session_notification(notification).await
                                    {
                                        error!("Failed to send session notification: {:?}", e);
                                    }
                                    continue;
                                }
                            }
                            AmpTool::CreateFile => {
                                let tool_call: Result<AmpCreateToolCall, serde_json::Error> =
                                    serde_json::from_value(tool_use_content_block.input.clone());
                                if let Ok(t) = tool_call {
                                    content.push(ToolCallContent::Content {
                                        content: ContentBlock::Text(TextContent {
                                            annotations: None,
                                            text: t.content,
                                            meta: None,
                                        }),
                                    });
                                }
                            }
                            AmpTool::Task => {
                                let tool_call: Result<AmpTaskToolCall, serde_json::Error> =
                                    serde_json::from_value(tool_use_content_block.input.clone());

                                if let Ok(tool_call) = &tool_call {
                                    tasks
                                        .entry(tool_use_content_block.id.clone())
                                        .or_insert(tool_call.clone());
                                }
                            }
                            _ => {}
                        }

                        let notification = SessionNotification {
                            session_id: session_id.clone(),
                            update: SessionUpdate::ToolCall(ToolCall {
                                id: ToolCallId(Arc::from(tool_use_content_block.id.clone())),
                                kind: amp_tool_to_tool_kind(&tool_use_content_block.name),
                                status: ToolCallStatus::Pending,
                                title,
                                content,
                                locations: vec![],
                                raw_input: None,
                                raw_output: None,

                                meta: None,
                            }),
                            meta: None,
                        };

                        if let Err(e) = self.client().session_notification(notification).await {
                            error!("Failed to send session notification: {:?}", e);
                        }
                    }
                    AmpContentBlock::ToolResult(tool_result_content_block) => {
                        let mut update = None;
                        let mut line = None;

                        //check if theres a file edit for this tool call
                        if let Some(file_edit) =
                            file_edits.remove(&tool_result_content_block.tool_use_id)
                        {
                            if let Some(result) = &tool_result_content_block.run.get("result") {
                                // Parse the diff to get the line numbers
                                if let Some(diff) = result.get("diff") {
                                    if let Some(diff_str) = diff.as_str() {
                                        line = get_line_number_from_diff_str(diff_str);
                                    }
                                }
                            }
                            update = Some(ToolCallUpdate {
                                id: ToolCallId(Arc::from(
                                    tool_result_content_block.tool_use_id.clone(),
                                )),
                                fields: ToolCallUpdateFields {
                                    kind: None,
                                    status: Some(ToolCallStatus::Completed),
                                    title: None,
                                    content: Some(vec![ToolCallContent::Diff {
                                        diff: Diff {
                                            path: PathBuf::from(file_edit.path.clone()),
                                            old_text: file_edit.old_str,
                                            new_text: file_edit.new_str,
                                            meta: None,
                                        },
                                    }]),
                                    locations: Some(vec![ToolCallLocation {
                                        path: PathBuf::from(file_edit.path.clone()),
                                        line,
                                        meta: None,
                                    }]),
                                    raw_input: None,
                                    raw_output: None,
                                },
                                meta: None,
                            });
                        } else if tasks.contains_key(&tool_result_content_block.tool_use_id) {
                            if let Some(progress) = tool_result_content_block.run.get("progress") {
                                if let Some(thread_id) = progress.get("threadID") {
                                    dbg!("hi");
                                    loop {
                                        let conversation = match self.get_amp_thread(SessionId(
                                            Arc::from(thread_id.as_str().unwrap()),
                                        )) {
                                            Some(conversation) => conversation,
                                            None => continue,
                                        };
                                        if let Err(e) = self
                                            .client()
                                            .session_notification(SessionNotification {
                                                session_id: session_id.clone(),
                                                update: SessionUpdate::ToolCallUpdate(
                                                    ToolCallUpdate {
                                                        id: ToolCallId(Arc::from(
                                                            tool_result_content_block
                                                                .tool_use_id
                                                                .clone(),
                                                        )),
                                                        fields: ToolCallUpdateFields {
                                                            kind: None,
                                                            status: Some(ToolCallStatus::Completed),
                                                            title: None,
                                                            content: Some(vec![
                                                                ToolCallContent::Content {
                                                                    content: ContentBlock::Text(
                                                                        TextContent {
                                                                            text: conversation
                                                                                .pretty_print(),
                                                                            annotations: None,
                                                                            meta: None,
                                                                        },
                                                                    ),
                                                                },
                                                            ]),
                                                            locations: None,
                                                            raw_input: None,
                                                            raw_output: None,
                                                        },
                                                        meta: None,
                                                    },
                                                ),
                                                meta: None,
                                            })
                                            .await
                                        {
                                            error!("Failed to send session notification: {:?}", e);
                                        }

                                        //check if the task finished
                                        if let Some(state) =
                                            &conversation.messages.last().unwrap().state
                                        {
                                            dbg!(&state);
                                            if let Some(stop_reason) = &state.stop_reason {
                                                dbg!(&stop_reason);
                                                if stop_reason == "end_turn" {
                                                    dbg!("Stopping");
                                                    break;
                                                }
                                            }
                                        }
                                        sleep(Duration::from_millis(100)).await;
                                    }
                                }
                            }
                        } else {
                            update = Some(ToolCallUpdate {
                                id: ToolCallId(Arc::from(
                                    tool_result_content_block.tool_use_id.clone(),
                                )),
                                fields: ToolCallUpdateFields {
                                    content: Some(vec![ToolCallContent::Content {
                                        content: ContentBlock::Text(TextContent {
                                            text: tool_result_content_block.run.to_string(),
                                            annotations: None,
                                            meta: None,
                                        }),
                                    }]),
                                    kind: None,
                                    status: Some(ToolCallStatus::Completed),
                                    title: None,
                                    locations: None,
                                    raw_input: None,
                                    raw_output: None,
                                },

                                meta: None,
                            });
                        }
                        if let Some(update) = update {
                            if let Err(e) = self
                                .client()
                                .session_notification(SessionNotification {
                                    session_id: session_id.clone(),
                                    update: SessionUpdate::ToolCallUpdate(update),
                                    meta: None,
                                })
                                .await
                            {
                                error!("Failed to send session notification: {:?}", e);
                            }
                        };
                    }
                }
            }
        }
    }
}

fn get_line_number_from_diff_str(diff: &str) -> Option<u32> {
    let parts = diff.split("@@").collect::<Vec<&str>>();
    let header = parts.get(1)?.trim();
    let line_info_parts = header.split(" ").collect::<Vec<&str>>();
    let final_line_number = line_info_parts.get(1)?;
    let line_number_parts = final_line_number.split(",").collect::<Vec<&str>>();
    let line_number = line_number_parts.first()?.replace("+", "").parse::<u32>();

    line_number.ok()
}

#[async_trait::async_trait(?Send)]
impl Agent for AmpAgent {
    async fn initialize(&self, _request: InitializeRequest) -> Result<InitializeResponse, Error> {
        return Ok(InitializeResponse {
            meta: None,
            protocol_version: V1,
            agent_capabilities: AgentCapabilities {
                load_session: false,
                prompt_capabilities: PromptCapabilities {
                    image: false,
                    audio: false,
                    embedded_context: false,
                    meta: None,
                },
                mcp_capabilities: McpCapabilities {
                    http: false,
                    sse: false,
                    meta: None,
                },
                meta: None,
            },
            auth_methods: vec![],
        });
    }

    async fn authenticate(
        &self,
        _request: AuthenticateRequest,
    ) -> Result<AuthenticateResponse, Error> {
        Ok(AuthenticateResponse { meta: None })
    }

    async fn new_session(&self, request: NewSessionRequest) -> Result<NewSessionResponse, Error> {
        (*self.cwd).borrow_mut().replace(request.cwd.clone());

        Command::new("amp")
            .current_dir(request.cwd.clone())
            .args(["--version"])
            .output()
            .map_err(|_| {
                Error::invalid_request().with_data(
                    "Amp is not installed: curl -fsSL https://ampcode.com/install.sh | bash",
                )
            })?;

        let output = Command::new("amp")
            .current_dir(request.cwd.clone())
            .args(["threads", "new"])
            .output()
            .map_err(|e| Error::into_internal_error(e))?;

        let session_id = match String::from_utf8(output.stdout) {
            Ok(s) => Some(s.replace("\n", "")),
            Err(_) => None,
        };

        if let Some(session_id) = session_id {
            Ok(NewSessionResponse {
                session_id: SessionId(Arc::from(session_id)),
                modes: None,
                meta: None,
            })
        } else {
            Err(Error::internal_error())
        }
    }

    async fn load_session(
        &self,
        _request: LoadSessionRequest,
    ) -> Result<LoadSessionResponse, Error> {
        todo!()
        // Loading sessions is not currently suppored by Zed, the code below should be mostly what is needed to support this.
        // Note: There is an `if message.role != "user" {` that will need to be sorted out as session loading should replay user messages aswell
        //
        // if let Some(conversation) = self.get_amp_thread(request.session_id.clone()) {
        //     self.process_conversation(&conversation, request.session_id)
        //         .await;
        //     todo!()
        // } else {
        //     Err(Error::internal_error().with_data("Could not open amp thread"))
        // }
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        let prompt = request
            .prompt
            .iter()
            .map(|b| match b {
                ContentBlock::Text(text_content) => text_content.text.clone(),
                ContentBlock::Image(_) | ContentBlock::Audio(_) => String::new(),
                ContentBlock::ResourceLink(resource_link) => resource_link.uri.clone(),
                ContentBlock::Resource(embedded_resource) => match &embedded_resource.resource {
                    EmbeddedResourceResource::TextResourceContents(text_resource_contents) => {
                        text_resource_contents.text.clone()
                    }
                    EmbeddedResourceResource::BlobResourceContents(blob_resource_contents) => {
                        blob_resource_contents.blob.clone()
                    }
                },
            })
            .collect::<Vec<String>>()
            .join("");

        let mut child = Command::new("amp")
            .args(["threads", "continue", &request.session_id.0, "-x"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|_| Error::internal_error().with_data("Failed to start amp"))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).map_err(|e| {
                Error::internal_error().with_data(format!("Failed to send prompt to amp: {}", e))
            })?;
        }
        self.set_amp_command(child);

        // Implementation note
        // AMP has a json mode but this has some drawbacks
        // 1. Tokens within a message are not streamed
        // 2. No thinking
        // 3. Tool call and result blocks appear to come together rather then one by one
        //
        // Due to this we read the thread file directly and diff the changes. Although this is a more brittle and complicated approach it allows us to get the features laid out above which I believe provides a better user experience

        // We keep track of the state of the conversation so that we can diff it with the new state to know what to send to the acp client
        let mut conversation_so_far: Option<AmpConversation> = None;
        let session_id = request.session_id;

        let mut file_edits: HashMap<String, AmpEditFileToolCall> = HashMap::new();
        let mut tasks: HashMap<String, AmpTaskToolCall> = HashMap::new();
        loop {
            let res = (*self.amp_command)
                .borrow_mut()
                .as_mut()
                .unwrap()
                .try_wait();

            if res.is_err() {
                return Err(Error::internal_error());
            } else if let Ok(status) = res {
                let conversation = match self.get_amp_thread(session_id.clone()) {
                    Some(conversation) => conversation,
                    None => return Err(Error::internal_error()),
                };

                if conversation_so_far.is_none() {
                    conversation_so_far = Some(conversation.clone());
                } else if let Some(ref mut prev_conversation) = conversation_so_far {
                    let diff = prev_conversation.diff(&conversation);
                    if let Some(conversation) = diff {
                        self.process_conversation(
                            &conversation,
                            session_id.clone(),
                            &mut file_edits,
                            &mut tasks,
                        )
                        .await;
                    }
                    conversation_so_far = Some(conversation);

                    if status.is_some() {
                        // finished processing send a end turn response
                        return Ok(PromptResponse {
                            stop_reason: StopReason::EndTurn,
                            meta: None,
                        });
                    }
                }
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    async fn cancel(&self, _args: CancelNotification) -> Result<(), Error> {
        let res = (*self.amp_command).borrow_mut().as_mut().unwrap().kill();
        if res.is_err() {
            return Err(Error::internal_error().with_data("Could not kill the amp process"));
        }
        Ok(())
    }

    async fn set_session_mode(
        &self,
        _args: SetSessionModeRequest,
    ) -> Result<SetSessionModeResponse, Error> {
        todo!()
    }

    async fn ext_method(&self, _args: ExtRequest) -> Result<ExtResponse, Error> {
        Err(Error::method_not_found())
    }

    async fn ext_notification(&self, _args: ExtNotification) -> Result<(), Error> {
        Err(Error::method_not_found())
    }
}
