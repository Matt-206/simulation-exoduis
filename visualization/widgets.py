"""
Pygame UI widgets for the policy steering panel.
Checkbox toggles, float sliders, and section headers.
"""

import pygame
import numpy as np

COL_PANEL_BG = (18, 22, 32)
COL_WIDGET_BG = (30, 35, 50)
COL_WIDGET_BORDER = (85, 95, 115)
COL_WIDGET_ACTIVE = (100, 210, 255)
COL_WIDGET_KNOB = (255, 255, 255)
COL_WIDGET_TRACK = (45, 50, 68)
COL_WIDGET_FILL = (80, 175, 240)
COL_WIDGET_TEXT = (225, 230, 245)
COL_WIDGET_LABEL = (175, 185, 205)
COL_WIDGET_SECTION = (100, 210, 255)
COL_CHECK_ON = (80, 230, 120)
COL_CHECK_OFF = (90, 95, 115)


class Checkbox:
    def __init__(self, x, y, w, label, checked=False, key=None):
        self.rect = pygame.Rect(x, y, w, 24)
        self.box_rect = pygame.Rect(x, y + 2, 20, 20)
        self.label = label
        self.checked = checked
        self.key = key or label
        self.hovered = False

    def handle_click(self, mx, my):
        if self.rect.collidepoint(mx, my):
            self.checked = not self.checked
            return True
        return False

    def draw(self, surface, font):
        col = COL_CHECK_ON if self.checked else COL_CHECK_OFF
        pygame.draw.rect(surface, col, self.box_rect, border_radius=4)
        if self.checked:
            cx, cy = self.box_rect.center
            pygame.draw.line(surface, (255, 255, 255), (cx - 5, cy), (cx - 2, cy + 5), 3)
            pygame.draw.line(surface, (255, 255, 255), (cx - 2, cy + 5), (cx + 6, cy - 4), 3)
        else:
            pygame.draw.rect(surface, COL_WIDGET_BORDER, self.box_rect, 2, border_radius=4)

        lbl = font.render(self.label, True, COL_WIDGET_TEXT)
        surface.blit(lbl, (self.box_rect.right + 10, self.rect.y + 3))


class Slider:
    def __init__(self, x, y, w, label, value=0.5, vmin=0.0, vmax=1.0,
                 fmt=".2f", key=None):
        self.x = x
        self.y = y
        self.w = w
        self.label = label
        self.value = value
        self.vmin = vmin
        self.vmax = vmax
        self.fmt = fmt
        self.key = key or label
        self.dragging = False

        self.track_rect = pygame.Rect(x, y + 18, w, 12)
        self.full_rect = pygame.Rect(x, y, w, 36)

    def _val_to_x(self):
        frac = (self.value - self.vmin) / max(self.vmax - self.vmin, 1e-9)
        return self.x + int(frac * self.w)

    def _x_to_val(self, mx):
        frac = (mx - self.x) / max(self.w, 1)
        frac = max(0.0, min(1.0, frac))
        return self.vmin + frac * (self.vmax - self.vmin)

    def handle_mousedown(self, mx, my):
        expanded = self.full_rect.inflate(0, 10)
        if expanded.collidepoint(mx, my):
            self.dragging = True
            self.value = self._x_to_val(mx)
            return True
        return False

    def handle_mousemove(self, mx, my):
        if self.dragging:
            self.value = self._x_to_val(mx)
            return True
        return False

    def handle_mouseup(self):
        was = self.dragging
        self.dragging = False
        return was

    def draw(self, surface, font_label, font_value):
        # Label
        lbl = font_label.render(self.label, True, COL_WIDGET_LABEL)
        surface.blit(lbl, (self.x, self.y))

        # Value
        val_str = f"{self.value:{self.fmt}}"
        val_surf = font_value.render(val_str, True, COL_WIDGET_ACTIVE)
        surface.blit(val_surf, (self.x + self.w - val_surf.get_width(), self.y))

        # Track background with visible border
        pygame.draw.rect(surface, COL_WIDGET_TRACK, self.track_rect, border_radius=6)
        pygame.draw.rect(surface, COL_WIDGET_BORDER, self.track_rect, 1, border_radius=6)

        # Filled portion
        knob_x = self._val_to_x()
        fill_w = knob_x - self.x
        if fill_w > 2:
            fill_rect = pygame.Rect(self.x, self.track_rect.y, fill_w, 12)
            pygame.draw.rect(surface, COL_WIDGET_FILL, fill_rect, border_radius=6)

        # Knob: big white circle with blue core
        ky = self.track_rect.centery
        if self.dragging:
            pygame.draw.circle(surface, COL_WIDGET_ACTIVE, (knob_x, ky), 12, 2)
        pygame.draw.circle(surface, COL_WIDGET_KNOB, (knob_x, ky), 9)
        pygame.draw.circle(surface, COL_WIDGET_FILL, (knob_x, ky), 5)


class SectionHeader:
    def __init__(self, x, y, w, label):
        self.x = x
        self.y = y
        self.w = w
        self.label = label

    def draw(self, surface, font):
        lbl = font.render(self.label, True, COL_WIDGET_SECTION)
        surface.blit(lbl, (self.x, self.y))
        ly = self.y + font.get_height() + 2
        pygame.draw.line(surface, COL_WIDGET_BORDER, (self.x, ly), (self.x + self.w, ly), 1)


class PolicyPanel:
    """
    Toggleable side panel with policy checkboxes and parameter sliders.
    Reads from / writes to the model's config in real time.
    """

    WIDTH = 310

    def __init__(self, screen_h, model):
        self.visible = False
        self.screen_h = screen_h
        self.model = model
        self.surface = pygame.Surface((self.WIDTH, screen_h))

        self.font_section = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_label = pygame.font.SysFont("Consolas", 12)
        self.font_value = pygame.font.SysFont("Consolas", 12, bold=True)

        pad = 15
        w = self.WIDTH - 2 * pad
        y = 15

        self.widgets = []

        # ── Title ──
        self.widgets.append(SectionHeader(pad, y, w, "POLICY STEERING"))
        y += 28

        # ── Toggles ──
        self.widgets.append(SectionHeader(pad, y, w, "Policy Toggles"))
        y += 22

        self.chk_ice = Checkbox(pad, y, w, "Enable ICE Act", model.config.policy.hq_relocation_active, "ice_act")
        self.widgets.append(self.chk_ice); y += 26

        self.chk_remote = Checkbox(pad, y, w, "Remote Work Boost", model.config.policy.remote_work_penetration > 0.35, "remote_boost")
        self.widgets.append(self.chk_remote); y += 26

        self.chk_school = Checkbox(pad, y, w, "School Subsidy", False, "school_subsidy")
        self.widgets.append(self.chk_school); y += 26

        self.chk_immigration = Checkbox(pad, y, w, "Immigration Active", model.config.policy.immigration_active, "immigration")
        self.widgets.append(self.chk_immigration); y += 26

        self.chk_shinkansen = Checkbox(pad, y, w, "Shinkansen Expansion", model.config.policy.shinkansen_expansion_active, "shinkansen")
        self.widgets.append(self.chk_shinkansen); y += 26

        self.chk_enterprise = Checkbox(pad, y, w, "Enterprise Zones", model.config.policy.enterprise_zones_active, "enterprise")
        self.widgets.append(self.chk_enterprise); y += 32

        # ── Economic Sliders ──
        self.widgets.append(SectionHeader(pad, y, w, "Economic Parameters"))
        y += 22

        self.sld_tokyo_tax = Slider(pad, y, w, "Tokyo Tax Premium", 500, 0, 2000, ".0f", "tokyo_tax")
        self.widgets.append(self.sld_tokyo_tax); y += 38

        self.sld_regional_wage = Slider(pad, y, w, "Regional Wage Mult", 1.0, 0.8, 1.5, ".2f", "regional_wage")
        self.widgets.append(self.sld_regional_wage); y += 38

        self.sld_childcare = Slider(pad, y, w, "Childcare Subsidy %", model.config.policy.childcare_subsidy_ratio, 0, 1.0, ".0%", "childcare")
        self.widgets.append(self.sld_childcare); y += 38

        self.sld_housing_peri = Slider(pad, y, w, "Housing Sub. Peri %", model.config.policy.housing_subsidy_periphery, 0, 0.5, ".0%", "housing_peri")
        self.widgets.append(self.sld_housing_peri); y += 38

        self.sld_remote_pct = Slider(pad, y, w, "Remote Work %", model.config.policy.remote_work_penetration, 0, 0.8, ".0%", "remote_pct")
        self.widgets.append(self.sld_remote_pct); y += 42

        # ── Psychological Weights ──
        self.widgets.append(SectionHeader(pad, y, w, "Psychological Weights"))
        y += 22

        self.sld_prestige = Slider(pad, y, w, "Prestige Sens. (wP)", model.config.weights.w_prestige, 0.05, 0.50, ".2f", "w_prestige")
        self.widgets.append(self.sld_prestige); y += 38

        self.sld_anomie = Slider(pad, y, w, "Anomie Weight (wA)", model.config.weights.w_anomie, 0.05, 0.50, ".2f", "w_anomie")
        self.widgets.append(self.sld_anomie); y += 38

        self.sld_friction = Slider(pad, y, w, "Friction Weight (wF)", model.config.weights.w_financial_friction, 0.10, 0.60, ".2f", "w_friction")
        self.widgets.append(self.sld_friction); y += 38

        self.sld_convenience = Slider(pad, y, w, "Convenience (wC)", model.config.weights.w_convenience, 0.05, 0.50, ".2f", "w_convenience")
        self.widgets.append(self.sld_convenience); y += 38

        self.sld_sqbias = Slider(pad, y, w, "Status Quo Bias", model.config.behavior.status_quo_bias, 0.05, 0.50, ".2f", "sq_bias")
        self.widgets.append(self.sld_sqbias); y += 42

        # ── Snapshot section ──
        self.widgets.append(SectionHeader(pad, y, w, "Snapshots"))
        y += 22
        self.snapshot_y = y
        self.snapshot_msg = ""
        self.snapshot_timer = 0

    def toggle(self):
        self.visible = not self.visible

    def handle_event(self, event, offset_x=0):
        if not self.visible:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            mx -= offset_x
            if mx < 0 or mx > self.WIDTH:
                return False
            for w in self.widgets:
                if isinstance(w, Checkbox) and w.handle_click(mx, my):
                    self._apply_toggles()
                    return True
                if isinstance(w, Slider) and w.handle_mousedown(mx, my):
                    return True
        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            mx -= offset_x
            for w in self.widgets:
                if isinstance(w, Slider) and w.handle_mousemove(mx, my):
                    self._apply_sliders()
                    return True
        elif event.type == pygame.MOUSEBUTTONUP:
            for w in self.widgets:
                if isinstance(w, Slider):
                    if w.handle_mouseup():
                        self._apply_sliders()
                        return True
        return False

    def _apply_toggles(self):
        cfg = self.model.config
        cfg.policy.hq_relocation_active = self.chk_ice.checked
        if self.chk_remote.checked:
            cfg.policy.remote_work_penetration = max(cfg.policy.remote_work_penetration, 0.40)
        cfg.policy.immigration_active = self.chk_immigration.checked
        cfg.policy.shinkansen_expansion_active = self.chk_shinkansen.checked
        cfg.policy.enterprise_zones_active = self.chk_enterprise.checked

        # School subsidy: prevent closure by raising the threshold to 0 (never triggers)
        if self.chk_school.checked:
            cfg.behavior.school_closure_pop_threshold = 0.0
        else:
            cfg.behavior.school_closure_pop_threshold = 0.30

    def _apply_sliders(self):
        cfg = self.model.config
        cfg.policy.circular_tax_per_child_monthly = self.sld_tokyo_tax.value
        cfg.policy.regional_wage_multiplier = self.sld_regional_wage.value
        cfg.policy.childcare_subsidy_ratio = self.sld_childcare.value
        cfg.policy.housing_subsidy_periphery = self.sld_housing_peri.value
        cfg.policy.remote_work_penetration = self.sld_remote_pct.value

        cfg.weights.w_prestige = self.sld_prestige.value
        cfg.weights.w_anomie = self.sld_anomie.value
        cfg.weights.w_financial_friction = self.sld_friction.value
        cfg.weights.w_convenience = self.sld_convenience.value
        cfg.behavior.status_quo_bias = self.sld_sqbias.value

        # Update compute engine weights live
        self.model.compute_engine.weights = cfg.weights
        self.model.compute_engine.behavior = cfg.behavior

    def show_snapshot_msg(self, msg):
        self.snapshot_msg = msg
        self.snapshot_timer = 180

    def draw(self):
        if not self.visible:
            return None

        self.surface.fill(COL_PANEL_BG)
        pygame.draw.line(self.surface, (55, 65, 85),
                         (self.WIDTH - 1, 0), (self.WIDTH - 1, self.screen_h), 2)

        for w in self.widgets:
            if isinstance(w, SectionHeader):
                w.draw(self.surface, self.font_section)
            elif isinstance(w, Checkbox):
                w.draw(self.surface, self.font_label)
            elif isinstance(w, Slider):
                w.draw(self.surface, self.font_label, self.font_value)

        # Snapshot instructions
        y = self.snapshot_y
        for line in ["F5: Save snapshot", "F6: Load snapshot"]:
            lbl = self.font_label.render(line, True, COL_WIDGET_LABEL)
            self.surface.blit(lbl, (15, y))
            y += 20

        if self.snapshot_timer > 0:
            self.snapshot_timer -= 1
            msg_surf = self.font_section.render(self.snapshot_msg, True, (80, 255, 120))
            self.surface.blit(msg_surf, (15, y + 5))

        # Hint at bottom
        hint_font = pygame.font.SysFont("Consolas", 11)
        hint = hint_font.render("Press P to hide this panel", True, (90, 95, 110))
        self.surface.blit(hint, (15, self.screen_h - 22))

        return self.surface
