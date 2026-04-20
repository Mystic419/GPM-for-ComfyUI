import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "gpm.gallery_browser.ui";
const NODE_NAME = "GPM Gallery Browser";
const MIN_VISIBLE_ROWS = 1;
const MAX_VISIBLE_ROWS = 6;
const DEFAULT_VISIBLE_ROWS = 3;
const DEFAULT_PROMPT_PROFILE = "SDXL";
const RANDOMIZE_OFF = "OFF";
const RANDOMIZE_ON = "ON";
const RANDOMIZE_OPTIONS = [RANDOMIZE_OFF, RANDOMIZE_ON];
const PROMPT_PROFILES = {
  SDXL: { personKey: "sdxl_person", sceneKey: "sdxl_scene" },
  Pony: { personKey: "pony_person", sceneKey: "pony_scene" },
  "Natural Language": { personKey: "natural_person", sceneKey: "natural_scene" },
};

const TILE_MIN_HEIGHT = 122;
const GRID_GAP = 8;
const GRID_VERTICAL_PADDING = 4;
const TOP_CONTROLS_HEIGHT = 30;
const ERROR_HEIGHT = 14;
const PROFILE_CONTROLS_HEIGHT = 30;
const BOTTOM_PANELS_HEIGHT = 110;
const ROOT_VERTICAL_PADDING = 16;
const SECTION_GAPS = 24;
const NODE_CHROME_HEIGHT = 44;
const SAVE_CLICK_TOKEN_PREFIX = "gpm-save";

function injectStylesOnce() {
  const styleId = "gpm-gallery-browser-style";
  if (document.getElementById(styleId)) {
    return;
  }

  const style = document.createElement("style");
  style.id = styleId;
  style.textContent = `
    .gpm-gallery-root {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 8px;
      box-sizing: border-box;
      background: #1f1f1f;
      border: 1px solid #333;
      border-radius: 6px;
      color: #ddd;
      height: 100%;
      overflow: hidden;
      font-size: 12px;
    }

    .gpm-gallery-top {
      display: flex;
      gap: 6px;
      align-items: center;
      flex: 0 0 auto;
    }

    .gpm-gallery-btn {
      border: 1px solid #4a4a4a;
      background: #2b2b2b;
      color: #ddd;
      border-radius: 4px;
      padding: 4px 8px;
      cursor: pointer;
      font-size: 12px;
    }

    .gpm-gallery-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .gpm-gallery-path {
      flex: 1;
      min-width: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      border: 1px solid #333;
      border-radius: 4px;
      padding: 4px 6px;
      background: #151515;
    }

    .gpm-gallery-grid {
      flex: 0 0 auto;
      min-height: 0;
      overflow: auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
      gap: 8px;
      align-content: start;
      padding: 2px;
      border: 1px solid #333;
      border-radius: 4px;
      background: #171717;
    }

    .gpm-gallery-tile {
      border: 1px solid #3a3a3a;
      border-radius: 4px;
      background: #222;
      padding: 4px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-height: 122px;
    }

    .gpm-gallery-tile:hover {
      border-color: #6a6a6a;
    }

    .gpm-gallery-tile.selected {
      border: 2px solid #ff2d2d;
      box-shadow: 0 0 0 1px rgba(255, 45, 45, 0.65), 0 0 12px rgba(255, 45, 45, 0.25);
      background: #2a1f1f;
    }

    .gpm-gallery-thumb {
      width: 100%;
      height: 96px;
      object-fit: contain;
      object-position: center;
      border-radius: 3px;
      background: #101010;
      border: 1px solid #2e2e2e;
    }

    .gpm-gallery-folder {
      width: 100%;
      height: 96px;
      border-radius: 3px;
      border: 1px solid #2e2e2e;
      background: #262626;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #bbb;
      font-size: 11px;
      text-align: center;
      padding: 4px;
      box-sizing: border-box;
    }

    .gpm-gallery-name {
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .gpm-gallery-bottom {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      height: 110px;
      min-height: 110px;
      flex: 0 0 110px;
    }

    .gpm-gallery-panel {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .gpm-gallery-label {
      font-size: 11px;
      color: #bbb;
    }

    .gpm-gallery-text {
      flex: 1;
      min-height: 84px;
      resize: none;
      border: 1px solid #333;
      border-radius: 4px;
      background: #151515;
      color: #ddd;
      padding: 6px;
      font-size: 12px;
      box-sizing: border-box;
    }

    .gpm-gallery-empty {
      color: #888;
      font-size: 12px;
      padding: 6px;
    }

    .gpm-gallery-error {
      color: #ff7f7f;
      font-size: 11px;
      min-height: 14px;
      flex: 0 0 auto;
    }
  `;
  document.head.appendChild(style);
}

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function hideWidget(widget) {
  if (!widget) return;
  widget.type = "converted-widget";
  widget.hidden = true;
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
}

function clampVisibleRows(value) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_VISIBLE_ROWS;
  }
  return Math.min(MAX_VISIBLE_ROWS, Math.max(MIN_VISIBLE_ROWS, parsed));
}

function calculateGridHeight(rows) {
  return (rows * TILE_MIN_HEIGHT) + ((rows - 1) * GRID_GAP) + GRID_VERTICAL_PADDING;
}

function calculateNodeHeight(rows) {
  const gridHeight = calculateGridHeight(rows);
  const contentHeight = TOP_CONTROLS_HEIGHT + gridHeight + ERROR_HEIGHT + PROFILE_CONTROLS_HEIGHT + BOTTOM_PANELS_HEIGHT + ROOT_VERTICAL_PADDING + SECTION_GAPS;
  return contentHeight + NODE_CHROME_HEIGHT;
}

function emptyProfilePrompts() {
  const result = {};
  for (const profileName of Object.keys(PROMPT_PROFILES)) {
    result[profileName] = { person: "", scene: "" };
  }
  return result;
}

function normalizePromptProfile(value) {
  return Object.prototype.hasOwnProperty.call(PROMPT_PROFILES, value) ? value : DEFAULT_PROMPT_PROFILE;
}

function normalizeRandomizeMode(value) {
  return RANDOMIZE_OPTIONS.includes(value) ? value : RANDOMIZE_OFF;
}

function getProfilePrompts(state, profileName) {
  if (!state.prompts[profileName]) {
    state.prompts[profileName] = { person: "", scene: "" };
  }
  return state.prompts[profileName];
}

app.registerExtension({
  name: EXTENSION_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    const onResize = nodeType.prototype.onResize;
    const onConfigure = nodeType.prototype.onConfigure;
    const onExecuted = nodeType.prototype.onExecuted;

    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      injectStylesOnce();
      if (this?.id !== undefined && this?.id !== null) {
        this.__gpmNodeId = String(this.id);
      }

      const rootWidget = findWidget(this, "root_folder");
      const currentWidget = findWidget(this, "current_subfolder");
      const selectedWidget = findWidget(this, "selected_image_rel");
      const visibleRowsWidget = findWidget(this, "visible_rows");
      const personPromptWidget = findWidget(this, "person_prompt_text");
      const scenePromptWidget = findWidget(this, "scene_prompt_text");
      const promptProfileWidget = findWidget(this, "prompt_profile");
      const randomizeModeWidget = findWidget(this, "randomize_mode");

      hideWidget(rootWidget);
      hideWidget(currentWidget);
      hideWidget(selectedWidget);
      hideWidget(visibleRowsWidget);
      hideWidget(personPromptWidget);
      hideWidget(scenePromptWidget);
      hideWidget(promptProfileWidget);
      hideWidget(randomizeModeWidget);

      const state = {
        rootFolder: String(rootWidget?.value || ""),
        currentSubfolder: String(currentWidget?.value || ""),
        selectedImageRel: String(selectedWidget?.value || ""),
        visibleRows: clampVisibleRows(visibleRowsWidget?.value ?? DEFAULT_VISIBLE_ROWS),
        items: [],
        promptProfile: normalizePromptProfile(String(promptProfileWidget?.value || DEFAULT_PROMPT_PROFILE)),
        randomizeMode: normalizeRandomizeMode(String(randomizeModeWidget?.value || RANDOMIZE_OFF)),
        prompts: emptyProfilePrompts(),
        promptCache: {},
        error: "",
        saveSequence: 0,
      };
      let saveInFlight = false;
      getProfilePrompts(state, state.promptProfile).person = String(personPromptWidget?.value || "");
      getProfilePrompts(state, state.promptProfile).scene = String(scenePromptWidget?.value || "");

      const setWidgets = () => {
        if (rootWidget) rootWidget.value = state.rootFolder;
        if (currentWidget) currentWidget.value = state.currentSubfolder;
        if (selectedWidget) selectedWidget.value = state.selectedImageRel;
        if (visibleRowsWidget) visibleRowsWidget.value = state.visibleRows;
        if (promptProfileWidget) promptProfileWidget.value = state.promptProfile;
        if (randomizeModeWidget) randomizeModeWidget.value = state.randomizeMode;
        const activePrompts = getProfilePrompts(state, state.promptProfile);
        if (personPromptWidget) personPromptWidget.value = activePrompts.person;
        if (scenePromptWidget) scenePromptWidget.value = activePrompts.scene;
      };

      const getNodeId = () => {
        const normalizeNodeId = (value) => {
          if (value === undefined || value === null) return "";
          const text = String(value).trim();
          if (!text) return "";
          const parsed = Number.parseInt(text, 10);
          if (Number.isFinite(parsed) && String(parsed) === text && parsed < 0) return "";
          return text;
        };

        const configuredId = normalizeNodeId(this?.__gpmNodeId);
        if (configuredId) return configuredId;
        return normalizeNodeId(this?.id);
      };

      const waitForNodeId = async () => {
        for (let attempt = 0; attempt < 120; attempt += 1) {
          const nodeId = getNodeId();
          if (nodeId) return nodeId;
          await new Promise((resolve) => setTimeout(resolve, 50));
        }
        return "";
      };

      const hasWidgetRestoreState = () => Boolean(state.rootFolder && String(state.rootFolder).trim().length > 0);

      const persistState = async () => {
        try {
          const nodeId = await waitForNodeId();
          if (!nodeId) {
            console.warn("[GPM][persist] skip save: missing valid node id");
            return;
          }
          await api.fetchApi("/gpm/gallery/state", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              node_id: nodeId,
              root_folder: state.rootFolder,
              current_subfolder: state.currentSubfolder,
              selected_image_rel: state.selectedImageRel,
              visible_rows: state.visibleRows,
            }),
          });
        } catch {
          console.warn("[GPM][persist] failed to save state");
        }
      };

      const loadPersistedState = async () => {
        try {
          const nodeId = await waitForNodeId();
          if (!nodeId) {
            console.warn("[GPM][restore] skip JSON fallback: missing valid node id");
            return;
          }
          const response = await api.fetchApi(`/gpm/gallery/state?node_id=${encodeURIComponent(nodeId)}`);
          const payload = await response.json();
          if (typeof payload.root_folder === "string" && payload.root_folder.trim().length > 0) {
            state.rootFolder = payload.root_folder;
          }
          if (typeof payload.current_subfolder === "string") {
            state.currentSubfolder = payload.current_subfolder;
          }
          if (typeof payload.selected_image_rel === "string") {
            state.selectedImageRel = payload.selected_image_rel;
          }
          if (payload.visible_rows !== undefined && payload.visible_rows !== null) {
            state.visibleRows = clampVisibleRows(payload.visible_rows);
          }
          setWidgets();
        } catch {
          console.warn("[GPM][restore] failed to load persisted state");
        }
      };

      const rootEl = document.createElement("div");
      rootEl.className = "gpm-gallery-root";

      const topEl = document.createElement("div");
      topEl.className = "gpm-gallery-top";

      const backBtn = document.createElement("button");
      backBtn.className = "gpm-gallery-btn";
      backBtn.textContent = "Back";

      const selectRootBtn = document.createElement("button");
      selectRootBtn.className = "gpm-gallery-btn";
      selectRootBtn.textContent = "Select Root Folder";

      const rowsLabel = document.createElement("div");
      rowsLabel.className = "gpm-gallery-label";
      rowsLabel.textContent = "Rows";

      const rowsSelect = document.createElement("select");
      rowsSelect.className = "gpm-gallery-btn";
      rowsSelect.style.padding = "4px 6px";
      for (let row = MIN_VISIBLE_ROWS; row <= MAX_VISIBLE_ROWS; row += 1) {
        const option = document.createElement("option");
        option.value = String(row);
        option.textContent = String(row);
        rowsSelect.appendChild(option);
      }

      const randomizeLabel = document.createElement("div");
      randomizeLabel.className = "gpm-gallery-label";
      randomizeLabel.textContent = "Randomize";

      const randomizeSelect = document.createElement("select");
      randomizeSelect.className = "gpm-gallery-btn";
      randomizeSelect.style.padding = "4px 6px";
      for (const mode of RANDOMIZE_OPTIONS) {
        const option = document.createElement("option");
        option.value = mode;
        option.textContent = mode;
        randomizeSelect.appendChild(option);
      }

      const pathEl = document.createElement("div");
      pathEl.className = "gpm-gallery-path";

      topEl.appendChild(backBtn);
      topEl.appendChild(selectRootBtn);
      topEl.appendChild(rowsLabel);
      topEl.appendChild(rowsSelect);
      topEl.appendChild(randomizeLabel);
      topEl.appendChild(randomizeSelect);
      topEl.appendChild(pathEl);

      const gridEl = document.createElement("div");
      gridEl.className = "gpm-gallery-grid";

      const errorEl = document.createElement("div");
      errorEl.className = "gpm-gallery-error";

      const controlsEl = document.createElement("div");
      controlsEl.className = "gpm-gallery-top";

      const profileLabel = document.createElement("div");
      profileLabel.className = "gpm-gallery-label";
      profileLabel.textContent = "Prompt Profile";

      const profileSelect = document.createElement("select");
      profileSelect.className = "gpm-gallery-btn";
      profileSelect.style.padding = "4px 6px";
      for (const profileName of Object.keys(PROMPT_PROFILES)) {
        const option = document.createElement("option");
        option.value = profileName;
        option.textContent = profileName;
        profileSelect.appendChild(option);
      }

      const saveJsonBtn = document.createElement("button");
      saveJsonBtn.className = "gpm-gallery-btn";
      saveJsonBtn.textContent = "Save to JSON";

      controlsEl.appendChild(profileLabel);
      controlsEl.appendChild(profileSelect);
      controlsEl.appendChild(saveJsonBtn);

      const bottomEl = document.createElement("div");
      bottomEl.className = "gpm-gallery-bottom";

      const personPanel = document.createElement("div");
      personPanel.className = "gpm-gallery-panel";
      const personLabel = document.createElement("div");
      personLabel.className = "gpm-gallery-label";
      personLabel.textContent = "Person Prompt";
      const personText = document.createElement("textarea");
      personText.className = "gpm-gallery-text";
      personPanel.appendChild(personLabel);
      personPanel.appendChild(personText);

      const scenePanel = document.createElement("div");
      scenePanel.className = "gpm-gallery-panel";
      const sceneLabel = document.createElement("div");
      sceneLabel.className = "gpm-gallery-label";
      sceneLabel.textContent = "Scene Prompt";
      const sceneText = document.createElement("textarea");
      sceneText.className = "gpm-gallery-text";
      scenePanel.appendChild(sceneLabel);
      scenePanel.appendChild(sceneText);

      bottomEl.appendChild(personPanel);
      bottomEl.appendChild(scenePanel);

      rootEl.appendChild(topEl);
      rootEl.appendChild(gridEl);
      rootEl.appendChild(errorEl);
      rootEl.appendChild(controlsEl);
      rootEl.appendChild(bottomEl);

      const domWidget = this.addDOMWidget("gpm_gallery_browser", "div", rootEl, {
        serialize: false,
        hideOnZoom: false,
      });

      const applyRowsLayout = (resizeNode = true) => {
        const rows = clampVisibleRows(state.visibleRows);
        state.visibleRows = rows;
        const gridHeight = calculateGridHeight(rows);
        gridEl.style.height = `${gridHeight}px`;
        gridEl.style.minHeight = `${gridHeight}px`;
        gridEl.style.maxHeight = `${gridHeight}px`;
        rowsSelect.value = String(rows);

        const desiredHeight = calculateNodeHeight(rows);
        this.__gpmDesiredHeight = desiredHeight;
        domWidget.computeSize = (width) => [width, this.__gpmDesiredHeight];

        const contentHeight = desiredHeight - NODE_CHROME_HEIGHT;
        rootEl.style.height = `${contentHeight}px`;
        rootEl.style.minHeight = `${contentHeight}px`;
        rootEl.style.maxHeight = `${contentHeight}px`;

        if (!resizeNode || !this.setSize || !Array.isArray(this.size)) {
          return;
        }

        const width = this.size[0];
        if (!Number.isFinite(width)) {
          return;
        }

        if (Math.round(this.size[1]) === Math.round(desiredHeight)) {
          return;
        }

        this.__gpmApplyingSize = true;
        this.setSize([width, desiredHeight]);
      };

      const parentSubfolder = (value) => {
        if (!value) return "";
        const parts = value.split("/").filter((p) => p.length > 0);
        if (parts.length <= 1) return "";
        return parts.slice(0, -1).join("/");
      };

      const clearPrompts = () => {
        state.prompts = emptyProfilePrompts();
      };

      const updateStaticUi = () => {
        const currentPath = state.currentSubfolder || ".";
        pathEl.textContent = state.rootFolder ? `${state.rootFolder} / ${currentPath}` : "No root folder selected";
        rowsSelect.value = String(clampVisibleRows(state.visibleRows));
        profileSelect.value = state.promptProfile;
        randomizeSelect.value = state.randomizeMode;
        const activePrompts = getProfilePrompts(state, state.promptProfile);
        personText.value = activePrompts.person;
        sceneText.value = activePrompts.scene;
        errorEl.textContent = state.error || "";
      };

      const syncPromptFromEditors = () => {
        const activePrompts = getProfilePrompts(state, state.promptProfile);
        activePrompts.person = personText.value;
        activePrompts.scene = sceneText.value;
        setWidgets();
      };

      const imageUrl = (relPath) => {
        const root = encodeURIComponent(state.rootFolder);
        const rel = encodeURIComponent(relPath);
        return `/gpm/gallery/image?root_folder=${root}&image_rel_path=${rel}`;
      };

      const getVisibleImageItems = () => state.items.filter((item) => item.type === "image");
      const saveActivePromptsToJson = async () => {
        const nodeId = await waitForNodeId();
        if (!nodeId) {
          console.warn("[GPM][save_json] skip save: missing valid node id");
          return;
        }

        syncPromptFromEditors();

        state.saveSequence += 1;
        const saveSequence = state.saveSequence;
        const saveTimestamp = new Date().toISOString();
        const clickToken = `${SAVE_CLICK_TOKEN_PREFIX}:${nodeId}:${Date.now()}:${saveSequence}`;

        const rootFolderAtClick = String(state.rootFolder || "");
        const selectedImageRelAtClick = String(state.selectedImageRel || "");
        const promptProfileAtClick = normalizePromptProfile(String(state.promptProfile || DEFAULT_PROMPT_PROFILE));
        const activePromptsAtClick = getProfilePrompts(state, promptProfileAtClick);
        activePromptsAtClick.person = String(personText.value || "");
        activePromptsAtClick.scene = String(sceneText.value || "");
        const personPromptAtClick = activePromptsAtClick.person;
        const scenePromptAtClick = activePromptsAtClick.scene;
        const lockAlreadyActive = saveInFlight;

        console.debug("[GPM][save_json] click", {
          click_token: clickToken,
          node_id: nodeId,
          selected_image_rel: selectedImageRelAtClick,
          prompt_profile: promptProfileAtClick,
          root_folder: rootFolderAtClick,
          current_subfolder: String(state.currentSubfolder || ""),
          lock_already_active: lockAlreadyActive,
          person_text_length: personPromptAtClick.length,
          scene_text_length: scenePromptAtClick.length,
          save_sequence: saveSequence,
          timestamp: saveTimestamp,
        });

        if (saveInFlight) {
          state.error = "Save already in progress.";
          updateStaticUi();
          return;
        }

        if (!rootFolderAtClick || !selectedImageRelAtClick) {
          state.error = "Missing selected image.";
          updateStaticUi();
          return;
        }

        state.rootFolder = rootFolderAtClick;
        state.selectedImageRel = selectedImageRelAtClick;
        state.promptProfile = promptProfileAtClick;
        const activePrompts = getProfilePrompts(state, promptProfileAtClick);
        activePrompts.person = personPromptAtClick;
        activePrompts.scene = scenePromptAtClick;
        setWidgets();

        saveInFlight = true;
        try {
          await persistState();
          const response = await api.fetchApi("/gpm/gallery/save_prompts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              click_token: clickToken,
              save_sequence: saveSequence,
              save_timestamp: saveTimestamp,
              node_id: nodeId,
              root_folder: rootFolderAtClick,
              image_rel_path: selectedImageRelAtClick,
              prompt_profile: promptProfileAtClick,
              person_prompt: personPromptAtClick,
              scene_prompt: scenePromptAtClick,
            }),
          });
          const payload = await response.json();
          if (payload?.ok) {
            if (!state.promptCache[selectedImageRelAtClick]) {
              state.promptCache[selectedImageRelAtClick] = emptyProfilePrompts();
            }
            state.promptCache[selectedImageRelAtClick][promptProfileAtClick] = {
              person: personPromptAtClick,
              scene: scenePromptAtClick,
            };
            state.error = payload.created ? "Saved prompts to new JSON." : "Saved prompts to JSON.";
          } else {
            state.error = typeof payload?.error === "string" ? payload.error : "Save failed.";
          }
        } catch {
          state.error = "Save failed.";
        } finally {
          saveInFlight = false;
        }

        updateStaticUi();
        app.graph.setDirtyCanvas(true, true);
      };

      const fetchPrompts = async (imageRelPath) => {
        if (!state.rootFolder || !imageRelPath) {
          return emptyProfilePrompts();
        }

        try {
          const root = encodeURIComponent(state.rootFolder);
          const rel = encodeURIComponent(imageRelPath);
          const response = await api.fetchApi(`/gpm/gallery/prompts?root_folder=${root}&image_rel_path=${rel}`);
          const payload = await response.json();
          const prompts = emptyProfilePrompts();
          for (const [profileName, keys] of Object.entries(PROMPT_PROFILES)) {
            prompts[profileName].person = typeof payload[keys.personKey] === "string" ? payload[keys.personKey] : "";
            prompts[profileName].scene = typeof payload[keys.sceneKey] === "string" ? payload[keys.sceneKey] : "";
          }
          console.debug("[GPM][prompts] fetched from disk", {
            selected_image_rel: imageRelPath,
            prompt_profile: state.promptProfile,
          });
          return prompts;
        } catch {
          return emptyProfilePrompts();
        }
      };

      const loadPrompts = async (imageRelPath) => {
        if (!state.rootFolder || !imageRelPath) {
          clearPrompts();
          return;
        }
        state.prompts = await fetchPrompts(imageRelPath);
      };

      const selectImageRel = async (imageRelPath, skipPersist = false) => {
        state.selectedImageRel = imageRelPath || "";
        await loadPrompts(state.selectedImageRel);
        setWidgets();
        if (!skipPersist) {
          await persistState();
        }
        renderGrid();
        updateStaticUi();
        app.graph.setDirtyCanvas(true, true);
      };

      const renderGrid = () => {
        gridEl.innerHTML = "";

        if (!state.items.length) {
          const empty = document.createElement("div");
          empty.className = "gpm-gallery-empty";
          empty.textContent = state.rootFolder ? "No folders or images in this folder." : "Select a root folder to begin.";
          gridEl.appendChild(empty);
          return;
        }

        for (const item of state.items) {
          const tile = document.createElement("div");
          tile.className = "gpm-gallery-tile";
          if (item.type === "image" && item.rel_path === state.selectedImageRel) {
            tile.classList.add("selected");
          }

          if (item.type === "folder") {
            const folderBox = document.createElement("div");
            folderBox.className = "gpm-gallery-folder";
            folderBox.textContent = "Folder";
            tile.appendChild(folderBox);
          } else {
            const image = document.createElement("img");
            image.className = "gpm-gallery-thumb";
            image.loading = "lazy";
            image.src = imageUrl(item.rel_path);
            tile.appendChild(image);
          }

          const name = document.createElement("div");
          name.className = "gpm-gallery-name";
          name.textContent = item.name;
          tile.appendChild(name);

          tile.addEventListener("click", async () => {
            if (item.type === "folder") {
              state.currentSubfolder = item.rel_path || "";
              state.selectedImageRel = "";
              clearPrompts();
              state.promptCache = {};
              setWidgets();
              await persistState();
              await refreshList();
              return;
            }
            const clickedRelPath = item.rel_path || "";
            const sameImageReselection = clickedRelPath === state.selectedImageRel;
            console.debug("[GPM][select] image click", {
              selected_image_rel: clickedRelPath,
              same_image_reselection: sameImageReselection,
              prompt_profile: state.promptProfile,
            });
            await selectImageRel(clickedRelPath);
          });

          gridEl.appendChild(tile);
        }
      };

      const refreshList = async () => {
        if (!state.rootFolder) {
          state.items = [];
          state.error = "";
          backBtn.disabled = true;
          state.promptCache = {};
          renderGrid();
          updateStaticUi();
          return;
        }

        try {
          const root = encodeURIComponent(state.rootFolder);
          const current = encodeURIComponent(state.currentSubfolder);
          const response = await api.fetchApi(`/gpm/gallery/list?root_folder=${root}&current_subfolder=${current}`);
          const payload = await response.json();

          if (!payload.ok) {
            state.items = [];
            state.error = payload.error || "Unable to list folder.";
            state.selectedImageRel = "";
            clearPrompts();
          } else {
            state.items = Array.isArray(payload.items) ? payload.items : [];
            state.rootFolder = typeof payload.root_folder === "string" ? payload.root_folder : state.rootFolder;
            state.currentSubfolder = typeof payload.current_subfolder === "string" ? payload.current_subfolder : "";
            state.error = "";

            const hasSelectedImage = state.items.some((item) => item.type === "image" && item.rel_path === state.selectedImageRel);
            if (!hasSelectedImage) {
              state.selectedImageRel = "";
              clearPrompts();
            } else if (state.selectedImageRel) {
              await loadPrompts(state.selectedImageRel);
            }

            const visibleImageSet = new Set(getVisibleImageItems().map((item) => item.rel_path));
            state.promptCache = Object.fromEntries(
              Object.entries(state.promptCache).filter(([relPath]) => visibleImageSet.has(relPath))
            );

          }

          backBtn.disabled = !state.currentSubfolder;
          setWidgets();
          renderGrid();
          updateStaticUi();
          await persistState();
          app.graph.setDirtyCanvas(true, true);
        } catch {
          state.items = [];
          state.error = "Unable to reach gallery backend.";
          backBtn.disabled = true;
          renderGrid();
          updateStaticUi();
        }
      };

      const firstUiValue = (uiPayload, key) => {
        if (!uiPayload || !Object.prototype.hasOwnProperty.call(uiPayload, key)) {
          return undefined;
        }
        const value = uiPayload[key];
        return Array.isArray(value) ? value[0] : value;
      };

      this.__gpmApplyExecutionUi = (executionMessage) => {
        const uiPayload = executionMessage?.ui;
        if (!uiPayload || typeof uiPayload !== "object") {
          return;
        }

        const rootFolder = firstUiValue(uiPayload, "gpm_root_folder");
        const currentSubfolder = firstUiValue(uiPayload, "gpm_current_subfolder");
        const items = firstUiValue(uiPayload, "gpm_items");
        const error = firstUiValue(uiPayload, "gpm_error");
        const selectedImageRel = firstUiValue(uiPayload, "gpm_selected_image_rel");
        const personPrompt = firstUiValue(uiPayload, "gpm_person_prompt");
        const scenePrompt = firstUiValue(uiPayload, "gpm_scene_prompt");
        const promptProfile = firstUiValue(uiPayload, "gpm_prompt_profile");
        const randomizeMode = firstUiValue(uiPayload, "gpm_randomize_mode");

        if (typeof rootFolder === "string") state.rootFolder = rootFolder;
        if (typeof currentSubfolder === "string") state.currentSubfolder = currentSubfolder;
        if (Array.isArray(items)) state.items = items;
        if (typeof error === "string") state.error = error;
        if (typeof selectedImageRel === "string") state.selectedImageRel = selectedImageRel;
        if (typeof promptProfile === "string") state.promptProfile = normalizePromptProfile(promptProfile);
        if (typeof randomizeMode === "string") state.randomizeMode = normalizeRandomizeMode(randomizeMode);

        const activePrompts = getProfilePrompts(state, state.promptProfile);
        if (typeof personPrompt === "string") activePrompts.person = personPrompt;
        if (typeof scenePrompt === "string") activePrompts.scene = scenePrompt;

        setWidgets();
        renderGrid();
        updateStaticUi();
        app.graph.setDirtyCanvas(true, true);
        void persistState();
      };
      selectRootBtn.addEventListener("click", async () => {
        const value = window.prompt("Enter root folder path", state.rootFolder || "");
        if (value === null) {
          return;
        }

        state.rootFolder = value.trim();
        state.currentSubfolder = "";
        state.selectedImageRel = "";
        clearPrompts();
        state.promptCache = {};
        setWidgets();
        await persistState();
        await refreshList();
      });

      backBtn.addEventListener("click", async () => {
        state.currentSubfolder = parentSubfolder(state.currentSubfolder);
        state.selectedImageRel = "";
        clearPrompts();
        state.promptCache = {};
        setWidgets();
        await persistState();
        await refreshList();
      });

      rowsSelect.addEventListener("change", async () => {
        state.visibleRows = clampVisibleRows(rowsSelect.value);
        setWidgets();
        applyRowsLayout(true);
        updateStaticUi();
        await persistState();
        app.graph.setDirtyCanvas(true, true);
      });

      randomizeSelect.addEventListener("change", async () => {
        state.randomizeMode = RANDOMIZE_OPTIONS.includes(randomizeSelect.value) ? randomizeSelect.value : RANDOMIZE_OFF;
        setWidgets();
        await persistState();
        renderGrid();
        updateStaticUi();
        app.graph.setDirtyCanvas(true, true);
      });

      saveJsonBtn.onclick = null;
      saveJsonBtn.onclick = async () => {
        await saveActivePromptsToJson();
      };
      profileSelect.addEventListener("change", async () => {
        syncPromptFromEditors();
        state.promptProfile = Object.prototype.hasOwnProperty.call(PROMPT_PROFILES, profileSelect.value)
          ? profileSelect.value
          : DEFAULT_PROMPT_PROFILE;
        if (state.selectedImageRel) {
          await loadPrompts(state.selectedImageRel);
        }
        setWidgets();
        updateStaticUi();
        app.graph.setDirtyCanvas(true, true);
      });

      personText.addEventListener("input", () => {
        syncPromptFromEditors();
        app.graph.setDirtyCanvas(true, true);
      });

      sceneText.addEventListener("input", () => {
        syncPromptFromEditors();
        app.graph.setDirtyCanvas(true, true);
      });

      setWidgets();
      applyRowsLayout(true);
      updateStaticUi();
      renderGrid();

      const restoreAndRefresh = async () => {
        if (!hasWidgetRestoreState()) {
          await loadPersistedState();
        }
        setWidgets();
        applyRowsLayout(true);
        updateStaticUi();
        await refreshList();
      };
      restoreAndRefresh();

      return result;
    };

    nodeType.prototype.onConfigure = function (info) {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      if (info && info.id !== undefined && info.id !== null) {
        this.__gpmNodeId = String(info.id);
      }
      hideWidget(findWidget(this, "root_folder"));
      hideWidget(findWidget(this, "current_subfolder"));
      hideWidget(findWidget(this, "selected_image_rel"));
      hideWidget(findWidget(this, "visible_rows"));
      hideWidget(findWidget(this, "person_prompt_text"));
      hideWidget(findWidget(this, "scene_prompt_text"));
      hideWidget(findWidget(this, "prompt_profile"));
      hideWidget(findWidget(this, "randomize_mode"));
      return result;
    };

    nodeType.prototype.onExecuted = function (message) {
      const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
      if (typeof this.__gpmApplyExecutionUi === "function") {
        this.__gpmApplyExecutionUi(message);
      }
      return result;
    };
    nodeType.prototype.onResize = function (size) {
      if (onResize) {
        onResize.apply(this, arguments);
      }

      if (this.__gpmApplyingSize) {
        this.__gpmApplyingSize = false;
        return;
      }

      if (!Array.isArray(size) || size.length < 2) {
        return;
      }

      const rowsWidget = findWidget(this, "visible_rows");
      const rows = clampVisibleRows(rowsWidget?.value ?? DEFAULT_VISIBLE_ROWS);
      const desiredHeight = calculateNodeHeight(rows);

      if (Math.round(size[1]) !== Math.round(desiredHeight) && this.setSize) {
        this.__gpmApplyingSize = true;
        this.setSize([size[0], desiredHeight]);
      }
    };
  },
});





































