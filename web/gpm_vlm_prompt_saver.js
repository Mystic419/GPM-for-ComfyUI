import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
console.log("[GPM VLM Prompt Saver] script loaded");

const EXTENSION_NAME = "gpm.vlm_prompt_saver.ui";
const NODE_NAME = "GPM VLM Prompt Saver";
const DEBUG_GPM_VLM_PROMPT_SAVER = false;
const DEBUG_GPM_VLM_PROMPT_SAVER_EXECUTION = true;
const LOADED_STATE_KEY = "gpm_vlm_prompt_saver_loaded";

function findWidget(node, name) {
  return node?.widgets?.find((w) => w?.name === name);
}

function firstUiValue(uiPayload, key) {
  if (!uiPayload || !Object.prototype.hasOwnProperty.call(uiPayload, key)) {
    return undefined;
  }
  const value = uiPayload[key];
  return Array.isArray(value) ? value[0] : value;
}

function syncWidgetElement(widget) {
  const element = widget?.element;
  if (!element) return;
  try {
    if ("value" in element) {
      element.value = widget.value;
    }
    if (typeof element.dispatchEvent === "function") {
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    }
  } catch (error) {
    console.warn("[GPM VLM Prompt Saver] widget element sync failed", error);
  }
}

function setWidgetValue(widget, value) {
  if (!widget) return;
  widget.value = value;
  if (typeof widget.callback === "function") {
    widget.callback(widget.value);
  }
  syncWidgetElement(widget);
}

function parseUseBanListValue(value) {
  if (value === "ON" || value === true) return "ON";
  if (value === "OFF" || value === false) return "OFF";
  return "";
}

function storeLoadedState(node, loaded) {
  if (!node || !loaded || !loaded.id) return false;
  node.gpmVlmPromptSaverLoaded = loaded;
  if (!node.properties || typeof node.properties !== "object") {
    node.properties = {};
  }
  node.properties[LOADED_STATE_KEY] = loaded;
  console.log(`[GPM VLM Prompt Saver] loaded source stored: ${loaded.id}`);
  return true;
}

function resolveLoadedState(node) {
  if (node?.gpmVlmPromptSaverLoaded?.id) {
    return node.gpmVlmPromptSaverLoaded;
  }
  const fromProps = node?.properties?.[LOADED_STATE_KEY];
  if (fromProps && typeof fromProps === "object" && fromProps.id) {
    node.gpmVlmPromptSaverLoaded = fromProps;
    return fromProps;
  }
  return null;
}

function populateSaverWidgets(node) {
  const loaded = resolveLoadedState(node);
  if (!loaded || !loaded.id) {
    return false;
  }

  const nameWidget = findWidget(node, "preset_name");
  const promptWidget = findWidget(node, "system_prompt");
  const useBanWidget = findWidget(node, "use_ban_list");
  const banListWidget = findWidget(node, "ban_list");

  if (typeof loaded.name === "string") setWidgetValue(nameWidget, loaded.name);
  if (typeof loaded.system_prompt === "string") setWidgetValue(promptWidget, loaded.system_prompt);
  const useBanValue = parseUseBanListValue(loaded.use_ban_list);
  if (useBanValue) setWidgetValue(useBanWidget, useBanValue);
  if (typeof loaded.ban_list === "string") setWidgetValue(banListWidget, loaded.ban_list);

  requestAnimationFrame(() => {
    if (typeof loaded.name === "string") setWidgetValue(nameWidget, loaded.name);
    if (typeof loaded.system_prompt === "string") setWidgetValue(promptWidget, loaded.system_prompt);
    if (useBanValue) setWidgetValue(useBanWidget, useBanValue);
    if (typeof loaded.ban_list === "string") setWidgetValue(banListWidget, loaded.ban_list);
    node?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    console.log(`[GPM VLM Prompt Saver] loaded source applied to widgets: ${loaded.id}`);
  });
  return true;
}

function loadedFromUiPayload(uiPayload) {
  const loadedId = firstUiValue(uiPayload, "gpm_vlm_prompt_saver_loaded_id");
  if (!loadedId) return null;
  return {
    id: String(loadedId),
    name: String(firstUiValue(uiPayload, "gpm_vlm_prompt_saver_loaded_name") || ""),
    system_prompt: String(firstUiValue(uiPayload, "gpm_vlm_prompt_saver_loaded_system_prompt") || ""),
    use_ban_list: String(firstUiValue(uiPayload, "gpm_vlm_prompt_saver_loaded_use_ban_list") || ""),
    ban_list: String(firstUiValue(uiPayload, "gpm_vlm_prompt_saver_loaded_ban_list") || ""),
  };
}

function loadedFromResultTuple(resultTuple) {
  if (!Array.isArray(resultTuple) || resultTuple.length < 3) return null;
  try {
    const payload = JSON.parse(String(resultTuple[2] || "{}"));
    if (!payload || typeof payload !== "object" || !payload.loaded_id) {
      return null;
    }
    const banList = Array.isArray(payload.loaded_ban_list) ? payload.loaded_ban_list.join("\n") : "";
    return {
      id: String(payload.loaded_id),
      name: String(payload.loaded_name || ""),
      system_prompt: String(payload.loaded_system_prompt || ""),
      use_ban_list: payload.loaded_use_ban_list ? "ON" : "OFF",
      ban_list: banList,
    };
  } catch {
    return null;
  }
}

function captureLoadedStateFromMessage(node, message) {
  const uiPayload = message?.ui;
  const fromUi = loadedFromUiPayload(uiPayload);
  if (fromUi) return fromUi;
  const fromResult = loadedFromResultTuple(message?.result);
  if (fromResult) return fromResult;
  return null;
}

function isSaverNode(node) {
  const type = String(node?.type || "");
  const comfyClass = String(node?.comfyClass || "");
  const title = String(node?.title || "");
  return type === NODE_NAME || comfyClass === NODE_NAME || title === NODE_NAME;
}

function nodeIdForRequest(node) {
  const value = node?.id;
  if (value === undefined || value === null || String(value).trim() === "") return "";
  return String(value);
}

async function loadSelectedPresetIntoSaverFields(node) {
  const presetWidget = findWidget(node, "load_preset");
  const presetId = String(presetWidget?.value || "").trim();
  const nodeId = nodeIdForRequest(node);
  if (!presetId || !nodeId) {
    console.warn("[GPM VLM Prompt Saver] load button needs a selected preset and node id");
    return;
  }

  try {
    const response = await api.fetchApi("/gpm/vlm/prompt-saver/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: nodeId, preset_id: presetId }),
    });
    const payload = await response.json();
    if (!response.ok || !payload?.ok || !payload?.preset) {
      throw new Error(String(payload?.error || "preset load failed"));
    }
    const preset = payload.preset;
    const loaded = {
      id: String(preset.id || presetId),
      name: String(preset.name || ""),
      system_prompt: String(preset.system_prompt || ""),
      use_ban_list: preset.use_ban_list ? "ON" : "OFF",
      ban_list: Array.isArray(preset.ban_list) ? preset.ban_list.join("\n") : "",
    };
    if (storeLoadedState(node, loaded)) {
      populateSaverWidgets(node);
    }
  } catch (error) {
    console.warn("[GPM VLM Prompt Saver] load button failed", error);
  }
}

function installLoadButton(node) {
  if (!isSaverNode(node) || node.__gpmVlmPromptSaverLoadButtonInstalled) return;
  node.addWidget?.("button", "load_selected_preset", null, () => {
    void loadSelectedPresetIntoSaverFields(node);
  }, { label: "Load selected preset into fields" });
  node.__gpmVlmPromptSaverLoadButtonInstalled = true;
}

function relevantExecutionPayload(detail) {
  const envelope = detail?.output ?? detail?.result ?? detail;
  const uiPayload = envelope?.ui ?? detail?.ui;
  const loadedIdFromUi = firstUiValue(uiPayload, "gpm_vlm_prompt_saver_loaded_id");
  if (loadedIdFromUi) {
    return { kind: "ui", loadedId: String(loadedIdFromUi), uiKeys: Object.keys(uiPayload || {}) };
  }

  const resultTuple = envelope?.result;
  if (Array.isArray(resultTuple) && resultTuple.length >= 3) {
    try {
      const payload = JSON.parse(String(resultTuple[2] || "{}"));
      if (payload?.loaded_id || payload?.action === "LOAD FROM SOURCE") {
        return {
          kind: "result",
          loadedId: String(payload?.loaded_id || ""),
          action: String(payload?.action || ""),
          ui_payload_emitted: Boolean(payload?.ui_payload_emitted),
        };
      }
    } catch {
      return null;
    }
  }
  return null;
}

app.registerExtension({
  name: EXTENSION_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData || nodeData.name !== NODE_NAME || !nodeType?.prototype || nodeType.prototype.__gpmVlmPromptSaverPatched) {
      return;
    }

    const originalOnExecuted = nodeType.prototype.onExecuted;
    const originalOnConfigure = nodeType.prototype.onConfigure;
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onExecuted = function (message) {
      const result = typeof originalOnExecuted === "function" ? originalOnExecuted.apply(this, arguments) : undefined;
      const loaded = captureLoadedStateFromMessage(this, message);
      if (loaded && storeLoadedState(this, loaded)) {
        populateSaverWidgets(this);
      } else if (DEBUG_GPM_VLM_PROMPT_SAVER) {
        console.log("[GPM VLM Prompt Saver][debug] no load state in execution message", message);
      }
      return result;
    };

    nodeType.prototype.onConfigure = function (info) {
      const result = typeof originalOnConfigure === "function" ? originalOnConfigure.apply(this, arguments) : undefined;
      if (info?.properties?.[LOADED_STATE_KEY]) {
        if (!this.properties || typeof this.properties !== "object") {
          this.properties = {};
        }
        this.properties[LOADED_STATE_KEY] = info.properties[LOADED_STATE_KEY];
        this.gpmVlmPromptSaverLoaded = info.properties[LOADED_STATE_KEY];
      }
      requestAnimationFrame(() => {
        populateSaverWidgets(this);
      });
      return result;
    };

    nodeType.prototype.onNodeCreated = function () {
      const result = typeof originalOnNodeCreated === "function" ? originalOnNodeCreated.apply(this, arguments) : undefined;
      installLoadButton(this);
      setTimeout(() => {
        populateSaverWidgets(this);
      }, 0);
      return result;
    };

    nodeType.prototype.__gpmVlmPromptSaverPatched = true;
    console.log("[GPM VLM Prompt Saver] hooked node def");
  },

  async setup() {
    if (api && typeof api.addEventListener === "function" && !window.__gpmVlmPromptSaverExecutedProbeBound) {
      api.addEventListener("executed", (event) => {
        if (!DEBUG_GPM_VLM_PROMPT_SAVER_EXECUTION) return;
        const detail = event?.detail ?? event;
        const relevant = relevantExecutionPayload(detail);
        if (!relevant) return;
        console.log("[GPM VLM Prompt Saver] executed event seen", relevant);
      });
      window.__gpmVlmPromptSaverExecutedProbeBound = true;
    }

    window.gpmDebugPromptSaverNodes = function () {
      return app.graph?._nodes?.filter((n) => isSaverNode(n)).map((n) => ({
        id: n.id,
        type: n.type,
        comfyClass: n.comfyClass,
        title: n.title,
        patchedProto: !!n.constructor?.prototype?.__gpmVlmPromptSaverPatched,
        widgets: n.widgets?.map((w) => ({ name: w.name, value: w.value })),
      })) || [];
    };

    window.gpmApplyPromptSaverLoadedState = function () {
      const nodes = app.graph?._nodes || [];
      for (const n of nodes) {
        if (isSaverNode(n)) {
          populateSaverWidgets(n);
        }
      }
    };
  },

  nodeCreated(node) {
    installLoadButton(node);
  },

  loadedGraphNode(node) {
    installLoadButton(node);
  },
});
