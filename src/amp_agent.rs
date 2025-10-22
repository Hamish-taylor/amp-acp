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
use std::ops::Deref;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpEditFileToolCall {
    pub path: Option<String>,
    pub old_str: Option<String>,
    pub new_str: Option<String>,
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
    #[serde(flatten)]
    pub content: AmpTool,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AmpToolResultContentBlock {
    #[serde(rename = "toolUseID")]
    pub tool_use_id: String,
    pub run: serde_json::Value,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct AmpPlanWriteToolCall {
    pub todos: Vec<AmpPlanTodo>,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub struct AmpPlanTodo {
    pub id: String,
    pub content: String,
    pub status: AmpPlanTodoStatus,
    pub priority: AmpPlanTodoPriority,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
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

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
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
                                .content
                                .to_title(&amp_tool_use_content_block.content)
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
                if a.id == b.id && a.content == b.content {
                    None
                } else {
                    Some(AmpContentBlock::ToolUse(AmpToolUseContentBlock {
                        id: b.id.clone(),
                        content: b.content.clone(),
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
#[serde(tag = "name", content = "input")]
pub enum AmpTool {
    Bash(AmpBashToolCall),
    #[serde(rename = "create_file")]
    CreateFile(AmpCreateToolCall),
    #[serde(rename = "edit_file")]
    EditFile(AmpEditFileToolCall),
    #[serde(rename = "finder")]
    Finder(Value),
    #[serde(rename = "glob")]
    Glob(Value),
    Grep(Value),
    #[serde(rename = "mermaid")]
    Mermaid(Value),
    #[serde(rename = "oracle")]
    Oracle(Value),
    Read(AmpReadToolCall),
    #[serde(rename = "read_mcp_resource")]
    ReadMcpResource(Value),
    #[serde(rename = "read_web_page")]
    ReadWebPage(AmpWebReadToolCall),
    Task(AmpTaskToolCall),
    #[serde(rename = "todo_read")]
    TodoRead(Value),
    #[serde(rename = "todo_write")]
    TodoWrite(AmpPlanWriteToolCall),
    #[serde(rename = "undo_edit")]
    UndoEdit(Value),
    #[serde(rename = "web_search")]
    WebSearch(AmpWebSearchToolCall),
    #[serde(untagged)]
    Other(Value),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpGrepToolResult {
    result: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpReadToolCall {
    path: Option<String>,
    read_range: Option<Vec<i32>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpCreateToolCall {
    path: Option<String>,
    content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpBashToolCall {
    cmd: Option<String>,
    cwd: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpWebSearchToolCall {
    query: Option<String>,
    max_results: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpWebReadToolCall {
    url: Option<String>,
    prompt: Option<String>,
    raw: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmpTaskToolCall {
    prompt: Option<String>,
    description: Option<String>,
}

impl AmpTool {
    fn to_title(&self, content: &AmpTool) -> String {
        match self {
            AmpTool::Oracle(_) => "Consulting the Oracle".to_string(),
            AmpTool::Read(content) => {
                if let Some(path) = &content.path {
                    let path = PathBuf::from(path);
                    let file_name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or_default();
                    if path.is_file() {
                        format!("Read [{}](file://{})", file_name, path.display())
                    } else {
                        format!("Read {}", path.display())
                    }
                } else {
                    "Reading file".to_string()
                }
            }
            AmpTool::ReadMcpResource(_) => "Read mcp resource".to_string(),
            AmpTool::ReadWebPage(content) => {
                if let Some(url) = &content.url {
                    format!("Reading {}", url)
                } else {
                    "Read webpage".to_string()
                }
            }
            AmpTool::Task(_) => "Sub agent".to_string(),
            AmpTool::TodoRead(_) => "Todo read".to_string(),
            AmpTool::TodoWrite(_) => "Todo write".to_string(),
            AmpTool::UndoEdit(_) => "Undo edit".to_string(),
            AmpTool::WebSearch(content) => {
                if let Some(query) = &content.query {
                    format!("Searching for \"{}\"", query)
                } else {
                    "Web search".to_string()
                }
            }
            AmpTool::Other(_) => "Unknown".to_string(),
            AmpTool::Bash(content) => {
                if let Some(cmd) = &content.cmd {
                    cmd.clone()
                } else {
                    "Bash".to_string()
                }
            }
            AmpTool::CreateFile(content) => {
                if let Some(path) = &content.path {
                    let path = PathBuf::from(&path);
                    let file_name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or_default();

                    format!("Created [{}](file://{})", file_name, path.display())
                } else {
                    "Creating file".to_string()
                }
            }
            AmpTool::EditFile(content) => {
                if let Some(path) = &content.path {
                    let path = PathBuf::from(&path);
                    let file_name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or_default();
                    format!("Edited [{}](file://{})", file_name, path.display())
                } else {
                    "Editing file".to_string()
                }
            }
            AmpTool::Finder(_) => "Finder".to_string(),
            AmpTool::Glob(_) => "Glob".to_string(),
            AmpTool::Grep(_) => "Grep".to_string(),
            AmpTool::Mermaid(_) => "Mermaid".to_string(),
        }
    }
}

fn amp_tool_to_tool_kind(amp_tool: &AmpTool) -> ToolKind {
    match amp_tool {
        AmpTool::Bash(_) => ToolKind::Execute,
        AmpTool::CreateFile(_) => ToolKind::Edit,
        AmpTool::EditFile(_) => ToolKind::Edit,
        AmpTool::Finder(_) => ToolKind::Search,
        AmpTool::Glob(_) => ToolKind::Execute,
        AmpTool::Grep(_) => ToolKind::Execute,
        AmpTool::Mermaid(_) => ToolKind::Other,
        AmpTool::Oracle(_) => ToolKind::Think,
        AmpTool::Read(_) => ToolKind::Read,
        AmpTool::ReadMcpResource(_) => ToolKind::Fetch,
        AmpTool::ReadWebPage(_) => ToolKind::Fetch,
        AmpTool::Task(_) => ToolKind::Think,
        AmpTool::TodoRead(_) => ToolKind::Think,
        AmpTool::TodoWrite(_) => ToolKind::Think,
        AmpTool::UndoEdit(_) => ToolKind::Edit,
        AmpTool::WebSearch(_) => ToolKind::Search,
        AmpTool::Other(_) => ToolKind::Other,
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

        match serde_json::from_str(&contents) {
            Ok(conversation) => Some(conversation),
            Err(e) => panic!("Failed to parse AMP thread: {}", e),
        }
    }

    async fn process_conversation(
        &self,
        conversation: &AmpConversation,
        session_id: SessionId,
        tool_calls: &mut HashMap<String, AmpTool>,
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
                            .content
                            .to_title(&tool_use_content_block.content);
                        let mut content = vec![];

                        tool_calls.insert(
                            tool_use_content_block.id.clone(),
                            tool_use_content_block.content.clone(),
                        );
                        match &tool_use_content_block.content {
                            AmpTool::TodoWrite(input) => {
                                let notification = SessionNotification {
                                    session_id: session_id.clone(),
                                    update: SessionUpdate::Plan(input.to_acp_plan()),
                                    meta: None,
                                };

                                if let Err(e) =
                                    self.client().session_notification(notification).await
                                {
                                    error!("Failed to send session notification: {:?}", e);
                                }
                                continue;
                            }
                            AmpTool::CreateFile(input) => {
                                if let Some(text) = input.content.clone() {
                                    content.push(ToolCallContent::Content {
                                        content: ContentBlock::Text(TextContent {
                                            annotations: None,
                                            text,
                                            meta: None,
                                        }),
                                    });
                                }
                            }
                            _ => {}
                        }

                        let notification = SessionNotification {
                            session_id: session_id.clone(),
                            update: SessionUpdate::ToolCall(ToolCall {
                                id: ToolCallId(Arc::from(tool_use_content_block.id.clone())),
                                kind: amp_tool_to_tool_kind(&tool_use_content_block.content),
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
                        dbg!(&tool_result_content_block.tool_use_id);
                        dbg!(&tool_calls);
                        let tool = tool_calls
                            .get(&tool_result_content_block.tool_use_id)
                            .unwrap();

                        let mut update = ToolCallUpdate {
                            id: ToolCallId(Arc::from(
                                tool_result_content_block.tool_use_id.clone(),
                            )),
                            fields: ToolCallUpdateFields {
                                kind: None,
                                status: Some(ToolCallStatus::Completed),
                                title: None,
                                content: None,
                                locations: None,
                                raw_input: None,
                                raw_output: None,
                            },
                            meta: None,
                        };
                        match tool {
                            AmpTool::EditFile(amp_edit_file_tool_call) => {
                                if let Some(path) = &amp_edit_file_tool_call.path
                                    && let Some(new_str) = &amp_edit_file_tool_call.new_str
                                {
                                    let mut line = None;

                                    if let Some(result) =
                                        &tool_result_content_block.run.get("result")
                                    {
                                        // Parse the diff to get the line numbers
                                        if let Some(diff) = result.get("diff") {
                                            if let Some(diff_str) = diff.as_str() {
                                                line = get_line_number_from_diff_str(diff_str);
                                            }
                                        }
                                    }
                                    update.fields.content = Some(vec![ToolCallContent::Diff {
                                        diff: Diff {
                                            path: PathBuf::from(path.clone()),
                                            old_text: amp_edit_file_tool_call.old_str.clone(),
                                            new_text: new_str.clone(),
                                            meta: None,
                                        },
                                    }]);
                                    update.fields.locations = Some(vec![ToolCallLocation {
                                        path: PathBuf::from(path.clone()),
                                        line,
                                        meta: None,
                                    }]);
                                }
                            }
                            AmpTool::Task(_) => {
                                if let Some(progress) =
                                    tool_result_content_block.run.get("progress")
                                {
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
                                                                status: Some(
                                                                    ToolCallStatus::Completed,
                                                                ),
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
                                                error!(
                                                    "Failed to send session notification: {:?}",
                                                    e
                                                );
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
                            }
                            AmpTool::Grep(_) => {
                                //maybe we dont print this
                                if let Ok(result) = serde_json::from_value::<AmpGrepToolResult>(
                                    tool_result_content_block.run.clone(),
                                ) {
                                    update.fields.content = Some(vec![ToolCallContent::Content {
                                        content: ContentBlock::Text(TextContent {
                                            text: result
                                                .result
                                                .unwrap_or_default()
                                                .iter()
                                                .map(|result| {
                                                    let mut parts = result.split(":");
                                                    let path = PathBuf::from(
                                                        parts.next().unwrap_or_default(),
                                                    );
                                                    format!(
                                                        "[{}](file://{}) `{} {}` \n\n",
                                                        path.file_name()
                                                            .unwrap_or_default()
                                                            .to_str()
                                                            .unwrap_or_default(),
                                                        path.display(),
                                                        parts.next().unwrap_or_default(),
                                                        parts.next().unwrap_or_default()
                                                    )
                                                })
                                                .collect(),
                                            annotations: None,
                                            meta: None,
                                        }),
                                    }]);
                                }
                            }
                            AmpTool::Finder(_)
                            | AmpTool::Glob(_)
                            | AmpTool::Mermaid(_)
                            | AmpTool::Oracle(_)
                            | AmpTool::Read(_)
                            | AmpTool::ReadMcpResource(_)
                            | AmpTool::ReadWebPage(_)
                            | AmpTool::Bash(_)
                            | AmpTool::CreateFile(_)
                            | AmpTool::TodoRead(_)
                            | AmpTool::TodoWrite(_)
                            | AmpTool::UndoEdit(_)
                            | AmpTool::WebSearch(_)
                            | AmpTool::Other(_) => {}
                        }

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

        let mut tool_calls: HashMap<String, AmpTool> = HashMap::new();
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
                            &mut tool_calls,
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
