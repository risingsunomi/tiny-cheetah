from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import List, Optional

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import Screen, ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static

from tiny_cheetah.tui.training_path_types import TrainingNode, NODE_STATUS_STYLES, NODE_STATUS_SYMBOLS


class TrainingPathScreen(Screen[Optional[List[TrainingNode]]]):
    """Dedicated editor for the sequential training path."""

    CSS_PATH = Path(__file__).with_name("training_path.tcss")

    BINDINGS = [
        Binding("escape", "cancel", "Back"),
    ]

    def __init__(self, nodes: List[TrainingNode]) -> None:
        super().__init__()
        self._path_nodes: List[TrainingNode] = copy.deepcopy(nodes) or [TrainingNode("Base Training")]
        self._ensure_base_step()
        self._path_list: Optional[ListView] = None
        self._graph_canvas: Optional[Container] = None
        self._feedback: Optional[Label] = None
        self._selected_index = 0
        self._drag_index: Optional[int] = None
        self._rename_button: Optional[Button] = None
        self._delete_button: Optional[Button] = None

    def compose(self) -> ComposeResult:
        items = [self._build_list_item(index, node) for index, node in enumerate(self._path_nodes)]
        yield Header(show_clock=True)
        with Container(id="path-root"):
            yield Label("Training Path Editor", id="path-title")
            with Container(id="path-body"):
                with VerticalScroll(id="path-graph-scroll"):
                    graph = Container(id="path-graph")
                    self._graph_canvas = graph
                    yield graph
                with Container(id="path-sidebar"):
                    list_view = ListView(*items, id="path-list")
                    self._path_list = list_view
                    yield list_view
                    feedback = Label("", id="path-feedback")
                    self._feedback = feedback
                    yield feedback
                    with Container(id="path-actions"):
                        yield Button("Add Step", id="path-add", variant="primary")
                        rename_btn = Button("Rename Step", id="path-rename")
                        self._rename_button = rename_btn
                        yield rename_btn
                        delete_btn = Button("Delete Step", id="path-delete", variant="error")
                        self._delete_button = delete_btn
                        yield delete_btn
                        yield Button("Reset", id="path-reset", variant="warning")
                        yield Button("Cancel", id="path-cancel")
                        yield Button("Save & Return", id="path-save", variant="success")
        yield Footer()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_mount(self) -> None:
        self._refresh_views()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "path-add": self._handle_add,
            "path-rename": lambda: self._handle_rename(self._selected_index),
            "path-delete": lambda: self._handle_delete(self._selected_index),
            "path-reset": self._handle_reset,
            "path-cancel": lambda: self.dismiss(None),
            "path-save": self._handle_save,
        }
        handler = handlers.get(event.button.id)
        if handler is not None:
            handler()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        index = self._index_from_item(event.item)
        if index is not None:
            self._selected_index = index
            self._refresh_graph()
            self._update_action_buttons()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = self._index_from_item(event.item)
        if index is None:
            return
        self._selected_index = index
        self._refresh_graph()
        self._update_action_buttons()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 1:
            node_index = self._graph_node_index_from_control(event.control)
            if node_index is not None:
                self._drag_index = node_index

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._drag_index is None:
            return
        target_index = self._graph_node_index_from_control(event.control)
        if target_index is not None and target_index != self._drag_index:
            self._move_node(self._drag_index, target_index)
        self._drag_index = None

    # ---- Event handlers -------------------------------------------------

    def _handle_add(self) -> None:
        modal = NodeStepModal()
        self.app.push_screen(modal, self._on_add_result)

    def _handle_rename(self, index: int) -> None:
        if not self._path_nodes:
            self._set_feedback("No steps to rename.")
            return
        if index <= 0:
            self._set_feedback("Base step name is derived from the model.")
            return
        current = self._path_nodes[index]
        modal = NodeStepModal(title="Rename Training Step", initial=current.name, confirm_label="Save")
        self.app.push_screen(modal, lambda result: self._on_rename_result(index, result))

    def _handle_delete(self, index: int) -> None:
        if len(self._path_nodes) <= 1 or index <= 0:
            self._set_feedback("Cannot delete the base step.")
            return
        node = self._path_nodes[index]
        modal = PathConfirmModal(
            f"Delete '{node.name}'?",
            "This step will be removed permanently.",
            confirm_label="Delete",
            confirm_variant="error",
        )
        self.app.push_screen(modal, lambda confirmed: self._on_delete_result(index, confirmed))

    def _handle_reset(self) -> None:
        modal = PathConfirmModal(
            "Reset Training Path?",
            "This restores the path to a single base training step.",
            confirm_label="Reset",
            confirm_variant="warning",
        )
        self.app.push_screen(modal, self._on_reset_confirmed)

    def _handle_save(self) -> None:
        for node in self._path_nodes:
            node.status = "pending"
        self._ensure_base_step()
        self.dismiss(copy.deepcopy(self._path_nodes))

    # ---- Results from modals -------------------------------------------

    def _on_add_result(self, result: Optional[str]) -> None:
        if result is None:
            return
        name = result.strip()
        if not name:
            self._set_feedback("Step name cannot be blank.")
            return
        base_settings = copy.deepcopy(self._path_nodes[0].settings)
        self._path_nodes.append(TrainingNode(name, settings=base_settings))
        self._selected_index = len(self._path_nodes) - 1
        self._refresh_views()
        self._set_feedback(f"Added step '{name}'.")

    def _on_rename_result(self, index: int, result: Optional[str]) -> None:
        if result is None:
            return
        name = result.strip()
        if not name:
            self._set_feedback("Step name cannot be blank.")
            return
        self._path_nodes[index].name = name
        self._refresh_views()
        self._set_feedback(f"Step renamed to '{name}'.")

    def _on_delete_result(self, index: int, confirmed: Optional[bool]) -> None:
        if not confirmed:
            return
        removed = self._path_nodes.pop(index)
        if self._selected_index >= len(self._path_nodes):
            self._selected_index = len(self._path_nodes) - 1
        self._refresh_views()
        self._set_feedback(f"Removed step '{removed.name}'.")

    def _on_reset_confirmed(self, confirmed: Optional[bool]) -> None:
        if not confirmed:
            return
        self._path_nodes = [TrainingNode("Base Training")]
        self._selected_index = 0
        self._refresh_views()
        self._set_feedback("Training path reset to base step.")

    # ---- Rendering ------------------------------------------------------

    def _refresh_views(self) -> None:
        asyncio.create_task(self._async_refresh_views())

    async def _async_refresh_views(self) -> None:
        await self._refresh_list()
        await self._refresh_graph()
        self._update_action_buttons()

    async def _refresh_list(self) -> None:
        if self._path_list is None:
            return
        await self._path_list.clear()
        for index, node in enumerate(self._path_nodes):
            await self._path_list.mount(self._build_list_item(index, node))
        if self._path_nodes:
            target = max(0, min(self._selected_index, len(self._path_nodes) - 1))
            try:
                self._path_list.index = target
            except AttributeError:
                pass

    async def _refresh_graph(self) -> None:
        if self._graph_canvas is None:
            return
        await self._graph_canvas.remove_children()
        if not self._path_nodes:
            await self._graph_canvas.mount(Static("No steps defined", classes="graph-empty"))
            return
        for index, node in enumerate(self._path_nodes):
            node_box = Static(
                self._format_node_label(node),
                classes=self._graph_classes(index, node),
                markup=True,
                id=f"graph-node-{index}",
            )
            await self._graph_canvas.mount(node_box)
            if index < len(self._path_nodes) - 1:
                connector = Static("│\n│\n▼", classes="graph-connector", markup=False)
                await self._graph_canvas.mount(connector)

    # ---- Helpers --------------------------------------------------------

    def _build_list_item(self, index: int, node: TrainingNode) -> ListItem:
        label = Label(self._format_list_label(index, node), markup=True)
        return ListItem(label, id=f"path-step-{index}")

    def _format_list_label(self, index: int, node: TrainingNode) -> str:
        symbol = NODE_STATUS_SYMBOLS.get(node.status, "•")
        color = NODE_STATUS_STYLES.get(node.status, "#bbbbbb")
        repeated = " (repeat)" if node.repeated else ""
        return f"[{color}]{symbol}[/{color}] Step {index + 1}: {node.name}{repeated}"

    def _format_node_label(self, node: TrainingNode) -> str:
        symbol = NODE_STATUS_SYMBOLS.get(node.status, "•")
        status = node.status.capitalize()
        repeated = "Repeat" if node.repeated else "Sequential"
        return f"[bold]{symbol} {node.name}[/]\n[dim]{status} · {repeated}[/]"

    def _graph_classes(self, index: int, node: TrainingNode) -> str:
        classes = ["graph-node", f"status-{node.status}"]
        if node.repeated:
            classes.append("repeat")
        if index == self._selected_index:
            classes.append("selected")
        return " ".join(classes)

    def _ensure_base_step(self) -> None:
        if not self._path_nodes:
            self._path_nodes.append(TrainingNode("Base Training"))
        else:
            self._path_nodes[0].name = self._path_nodes[0].name or "Base Training"
        self._path_nodes[0].status = (
            self._path_nodes[0].status if self._path_nodes[0].status in NODE_STATUS_STYLES else "pending"
        )
        self._path_nodes[0].repeated = False

    def _index_from_item(self, item: Optional[ListItem]) -> Optional[int]:
        if item is None or item.id is None:
            return None
        try:
            return int(item.id.split("-")[-1])
        except (ValueError, IndexError):
            return None

    def _set_feedback(self, message: str) -> None:
        if self._feedback is not None:
            self._feedback.update(message)

    def _graph_node_index_from_control(self, control: Optional[Widget]) -> Optional[int]:
        target = control
        while target is not None:
            if target.id and target.id.startswith("graph-node-"):
                try:
                    return int(target.id.split("-", 2)[2])
                except (IndexError, ValueError):
                    return None
            target = target.parent
        return None

    def _move_node(self, source: int, target: int) -> None:
        if source == target or target < 0 or target >= len(self._path_nodes):
            return
        if source == 0 or target == 0:
            self._set_feedback("Base step remains at the top.")
            return
        node = self._path_nodes.pop(source)
        target = max(0, min(target, len(self._path_nodes)))
        self._path_nodes.insert(target, node)
        self._selected_index = target
        self._refresh_views()
        self._set_feedback(f"Moved step to position {target + 1}.")
    def _update_action_buttons(self) -> None:
        allow_rename = self._selected_index > 0
        allow_delete = self._selected_index > 0 and len(self._path_nodes) > 1
        if self._rename_button is not None:
            self._rename_button.disabled = not allow_rename
        if self._delete_button is not None:
            self._delete_button.disabled = not allow_delete

class NodeStepModal(ModalScreen[Optional[str]]):
    """Modal dialog for capturing a training step name."""

    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(
        self,
        *,
        title: str = "Add Training Step",
        initial: str = "",
        confirm_label: str = "Add",
        placeholder: str = "e.g. Fine-tune dataset",
    ) -> None:
        super().__init__(id="node-step-modal")
        self._title = title
        self._initial = initial
        self._confirm_label = confirm_label
        self._placeholder = placeholder
        self._name_input: Optional[Input] = None

    def compose(self) -> ComposeResult:
        with Container(id="node-step-modal-container"):
            yield Label(self._title, id="node-step-title")
            self._name_input = Input(placeholder=self._placeholder, id="node-step-name")
            yield self._name_input
            with Container(id="node-step-buttons"):
                yield Button("Cancel", id="node-step-cancel")
                yield Button(self._confirm_label, id="node-step-confirm", variant="primary")

    def on_mount(self) -> None:
        if self._name_input is not None:
            self._name_input.value = self._initial
            self.call_after_refresh(self._name_input.focus)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "node-step-cancel":
            self.dismiss(None)
        elif event.button.id == "node-step-confirm":
            value = self._name_input.value.strip() if self._name_input else ""
            self.dismiss(value or None)


class PathConfirmModal(ModalScreen[Optional[bool]]):
    """Confirmation modal for destructive operations."""

    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(
        self,
        title: str,
        message: str,
        *,
        confirm_label: str = "Confirm",
        confirm_variant: str = "primary",
    ) -> None:
        super().__init__(id="path-confirm-modal")
        self._title = title
        self._message = message
        self._confirm_label = confirm_label
        self._confirm_variant = confirm_variant

    def compose(self) -> ComposeResult:
        with Container(id="path-confirm-modal-container"):
            yield Label(self._title, id="path-confirm-title")
            yield Static(self._message, id="path-confirm-message")
            with Container(id="path-confirm-buttons"):
                yield Button("Cancel", id="path-confirm-cancel")
                yield Button(self._confirm_label, id="path-confirm-accept", variant=self._confirm_variant)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "path-confirm-cancel":
            self.dismiss(False)
        elif event.button.id == "path-confirm-accept":
            self.dismiss(True)
