"""
pynics — Induced Current Density Explorer (Streamlit)
======================================================
Supports CASTEP current.dat and VASP NMRCURBX/Y/Z files.
Large files are parsed once and cached in the session; only the lightweight
slice plot JSON is recomputed on each slider interaction.

Run locally:
    pip install pynics[streamlit]
    streamlit run streamlit_app.py

Deploy:
    Push to GitHub and connect to Streamlit Community Cloud.
    Set Python version to 3.11 in Advanced settings.
"""

from __future__ import annotations

import io
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="pynics — Current Density Explorer",
    page_icon="⚛",
    layout="wide",
)

st.title("pynics — Induced Current Density Explorer")
st.markdown(
    "Upload a **CASTEP** `current.dat` (+ `.cell`) or **VASP** `NMRCURBX/Y/Z` "
    "(+ `POSCAR`/`CONTCAR`) file to explore induced current density slices."
)

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(raw_bytes: bytes) -> str:
    """Return 'castep' or 'vasp' by sniffing the first 512 bytes."""
    try:
        raw_bytes[:512].decode("ascii")
        return "vasp"
    except UnicodeDecodeError:
        return "castep"


def _write_tmp(content_bytes: bytes, suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(content_bytes)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Cached loader — re-runs only when file bytes change
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Parsing current density file…")
def load_current_data(cur_bytes: bytes, struct_bytes: bytes):
    """Parse uploaded files and return (CurrentData, ase.Atoms)."""
    from pynics import CurrentData
    from ase.io import read as ase_read
    from ase.io.vasp import read_vasp

    fmt = _detect_format(cur_bytes)
    cur_path = _write_tmp(cur_bytes, ".dat" if fmt == "castep" else "")
    struct_path = _write_tmp(struct_bytes, ".cell" if fmt == "castep" else "")

    if fmt == "castep":
        cd = CurrentData.from_castep(struct_path, cur_path)
        atoms = ase_read(struct_path)
    else:
        cd = CurrentData.from_vasp(cur_path, poscar_path=struct_path)
        atoms = read_vasp(struct_path)

    return cd, atoms, fmt


# ---------------------------------------------------------------------------
# Slice computation (pure function — fast, no caching needed)
# ---------------------------------------------------------------------------

def compute_slice(
    cd,
    atoms,
    axis_idx: int,
    slice_pos: float,
    b_index: int,
    step: int,
    show_atoms: bool,
    show_bonds: bool,
):
    nx, ny, nz = cd.grid
    lat = cd.lattice

    grid_sizes = [nx, ny, nz]
    normal_size = grid_sizes[axis_idx]
    ia, ib = [i for i in range(3) if i != axis_idx]
    na, nb = grid_sizes[ia], grid_sizes[ib]

    slice_idx = min(int(np.clip(slice_pos, 0.0, 1.0) * normal_size), normal_size - 1)

    # Current density, real part only
    j = np.real(cd.get_current_density(b_index=b_index))

    slices = [slice(None)] * 4
    slices[axis_idx + 1] = slice_idx
    slices[ia + 1] = slice(None, None, step)
    slices[ib + 1] = slice(None, None, step)
    j_slice = j[tuple(slices)]  # (3, na//step, nb//step)

    # Coordinate system: fractional → scaled Å
    # scr_x[i,j] = ia_frac[i] * |lat[ia]|,  scr_y[i,j] = ib_frac[j] * |lat[ib]|
    # This is self-consistent for any cell geometry (orthogonal or not) because
    # we only ever use in-plane fractional coordinates — the normal component
    # never contributes and atoms / heatmap always align.
    norm_ia = float(np.linalg.norm(lat[ia]))
    norm_ib = float(np.linalg.norm(lat[ib]))
    e_ia = lat[ia] / norm_ia   # unit vector along ia lattice direction
    e_ib = lat[ib] / norm_ib   # unit vector along ib lattice direction

    ia_fracs = np.arange(0, na, step) / na
    ib_fracs = np.arange(0, nb, step) / nb
    IB, IA = np.meshgrid(ib_fracs, ia_fracs)  # each (na//step, nb//step)

    scr_x = IA * norm_ia   # (na//step, nb//step)
    scr_y = IB * norm_ib

    j_vec = np.stack([j_slice[0], j_slice[1], j_slice[2]], axis=-1)  # (..., 3)
    j_u1 = j_vec @ e_ia   # component along ia lattice direction
    j_u2 = j_vec @ e_ib   # component along ib lattice direction
    j_mag = np.sqrt(j_slice[0]**2 + j_slice[1]**2 + j_slice[2]**2)

    # Atoms near the slice plane — use scaled (fractional) positions throughout
    # so they are in the same coordinate system as the heatmap.
    atom_screen = None
    scaled_all = atoms.get_scaled_positions() if atoms is not None else None
    if atoms is not None and show_atoms and scaled_all is not None:
        dist = np.abs(scaled_all[:, axis_idx] - slice_idx / normal_size)
        dist = np.minimum(dist, 1.0 - dist)
        close_mask = dist < 0.15
        if close_mask.any():
            frac_close = scaled_all[close_mask]
            sym_close = [atoms.get_chemical_symbols()[i]
                         for i in np.where(close_mask)[0]]
            atom_screen = {
                "x": frac_close[:, ia] * norm_ia,
                "y": frac_close[:, ib] * norm_ib,
                "symbols": sym_close,
            }

    # Bond pairs — detect in Cartesian (MIC), display in scaled-Å
    bond_pairs = []
    if atoms is not None and atom_screen is not None and show_bonds and scaled_all is not None:
        from ase.geometry import find_mic
        all_pos = atoms.get_positions()
        all_sym = atoms.get_chemical_symbols()
        close_idx = [
            i for i in range(len(all_sym))
            if min(
                abs(scaled_all[i, axis_idx] - slice_idx / normal_size),
                1 - abs(scaled_all[i, axis_idx] - slice_idx / normal_size),
            ) < 0.15
        ]
        for ii, i in enumerate(close_idx):
            for j_idx in close_idx[ii + 1:]:
                diff, _ = find_mic(all_pos[j_idx] - all_pos[i], lat)
                if np.linalg.norm(diff) < 1.6:
                    bond_pairs.append((
                        (scaled_all[i, ia] * norm_ia,   scaled_all[i, ib] * norm_ib),
                        (scaled_all[j_idx, ia] * norm_ia, scaled_all[j_idx, ib] * norm_ib),
                    ))

    return scr_x, scr_y, j_u1, j_u2, j_mag, atom_screen, bond_pairs


# ---------------------------------------------------------------------------
# Build plotly figure
# ---------------------------------------------------------------------------

_ELEM_COLOURS = {
    "C": "#404040", "H": "#FFFFFF", "N": "#3050F8",
    "O": "#FF0D0D", "F": "#90E050", "S": "#FFFF30",
}


def _add_arrows_line(fig, scr_x, scr_y, j_u1, j_u2, arrow_scale, arrow_color, arrow_width, normalize):
    """Simple tail-to-tip line segments (current default)."""
    j_norm = np.sqrt(j_u1**2 + j_u2**2 + 1e-30)
    j_max = float(j_norm.max()) or 1.0
    scale = arrow_scale * ((scr_x.max() - scr_x.min()) / max(scr_x.shape))
    if normalize:
        ux = j_u1 / j_max * scale
        uy = j_u2 / j_max * scale
    else:
        ux = j_u1 / j_max * scale * (j_norm / j_max)
        uy = j_u2 / j_max * scale * (j_norm / j_max)

    arrow_x, arrow_y = [], []
    for i in range(scr_x.shape[0]):
        for k in range(scr_x.shape[1]):
            x0, y0 = float(scr_x[i, k]), float(scr_y[i, k])
            dx, dy = float(ux[i, k]), float(uy[i, k])
            arrow_x += [x0, x0 + dx, None]
            arrow_y += [y0, y0 + dy, None]

    fig.add_trace(go.Scatter(
        x=arrow_x, y=arrow_y, mode="lines",
        line=dict(color=arrow_color, width=arrow_width),
        name="Current j",
    ))


def _add_arrows_quiver(fig, scr_x, scr_y, j_u1, j_u2, arrow_scale, arrow_color, arrow_width, normalize, arrow_head_scale):
    """Plotly figure_factory quiver — V-shaped arrowheads that scale with zoom."""
    import plotly.figure_factory as ff

    j_norm = np.sqrt(j_u1**2 + j_u2**2 + 1e-30)
    j_max = float(j_norm.max()) or 1.0
    scale = arrow_scale * ((scr_x.max() - scr_x.min()) / max(scr_x.shape))
    if normalize:
        u = j_u1 / j_max
        v = j_u2 / j_max
    else:
        u = j_u1 / j_max * (j_norm / j_max)
        v = j_u2 / j_max * (j_norm / j_max)

    quiv = ff.create_quiver(
        scr_x.ravel(), scr_y.ravel(),
        u.ravel(), v.ravel(),
        scale=scale,
        arrow_scale=arrow_head_scale,
        line=dict(color=arrow_color, width=arrow_width),
        name="Current j",
    )
    for trace in quiv.data:
        fig.add_trace(trace)


def _add_arrows_magnitude(fig, scr_x, scr_y, j_u1, j_u2, j_mag, arrow_scale, arrow_width, normalize, colorscale):
    """Arrows coloured by |j| magnitude using a separate scatter per arrow is too slow;
    instead we use a single trace with NaN gaps and a matching heatmap coloraxis."""
    j_norm = np.sqrt(j_u1**2 + j_u2**2 + 1e-30)
    j_max = float(j_norm.max()) or 1.0
    mag_max = float(j_mag.max()) or 1.0
    scale = arrow_scale * ((scr_x.max() - scr_x.min()) / max(scr_x.shape))
    if normalize:
        ux = j_u1 / j_max * scale
        uy = j_u2 / j_max * scale
    else:
        ux = j_u1 / j_max * scale * (j_norm / j_max)
        uy = j_u2 / j_max * scale * (j_norm / j_max)

    # One scatter trace per arrow is prohibitive; use segmented line + invisible
    # marker coloured by magnitude to drive a discrete colour range.
    # We draw line segments coloured by bucketing into 20 magnitude quantiles.
    n_buckets = 20
    buckets = np.linspace(0, mag_max, n_buckets + 1)
    import plotly.colors as pc
    colours = pc.sample_colorscale(colorscale, [b / mag_max for b in buckets[:-1]])

    for b in range(n_buckets):
        lo, hi = buckets[b], buckets[b + 1]
        mask = (j_mag >= lo) & (j_mag < hi)
        if not mask.any():
            continue
        ax, ay = [], []
        ii, kk = np.where(mask)
        for i, k in zip(ii, kk):
            x0, y0 = float(scr_x[i, k]), float(scr_y[i, k])
            dx, dy = float(ux[i, k]), float(uy[i, k])
            ax += [x0, x0 + dx, None]
            ay += [y0, y0 + dy, None]
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode="lines",
            line=dict(color=colours[b], width=arrow_width),
            showlegend=(b == n_buckets - 1),
            name="Current j" if b == n_buckets - 1 else None,
            legendgroup="current",
        ))


def build_figure(
    scr_x, scr_y, j_u1, j_u2, j_mag, atom_screen, bond_pairs,
    axis_label: str, slice_pos: float,
    ia_label: str = "b", ib_label: str = "c",
    arrow_style: str = "lines",
    arrow_scale: float = 0.3,
    arrow_color: str = "white",
    arrow_width: int = 1,
    normalize: bool = True,
    arrow_head_scale: float = 0.3,
    arrow_colorscale: str = "Plasma",
    heatmap_colorscale: str = "Viridis",
    heatmap_opacity: float = 0.5,
    heatmap_clim_mode: str = "auto",
    heatmap_clim_lo: float = 0.0,
    heatmap_clim_hi: float = 1.0,
    lock_aspect: bool = True,
    plot_height: int = 700,
) -> go.Figure:
    fig = go.Figure()

    # Compute colour limits
    mag_max = float(j_mag.max()) or 1.0
    if heatmap_clim_mode == "fraction":
        zmin, zmax = heatmap_clim_lo * mag_max, heatmap_clim_hi * mag_max
    elif heatmap_clim_mode == "absolute":
        zmin, zmax = heatmap_clim_lo, heatmap_clim_hi
    else:  # auto
        zmin, zmax = None, None

    # Heatmap
    fig.add_trace(go.Heatmap(
        x=scr_x[:, 0], y=scr_y[0, :], z=j_mag.T,
        colorscale=heatmap_colorscale, opacity=heatmap_opacity,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title="|j| (a.u.)", x=1.02),
        name="|j|",
    ))

    # Arrows — delegated to style helpers
    if arrow_style == "quiver":
        _add_arrows_quiver(fig, scr_x, scr_y, j_u1, j_u2,
                           arrow_scale, arrow_color, arrow_width, normalize, arrow_head_scale)
    elif arrow_style == "magnitude":
        _add_arrows_magnitude(fig, scr_x, scr_y, j_u1, j_u2, j_mag,
                              arrow_scale, arrow_width, normalize, arrow_colorscale)
    else:  # "lines"
        _add_arrows_line(fig, scr_x, scr_y, j_u1, j_u2,
                         arrow_scale, arrow_color, arrow_width, normalize)

    # Bonds
    for p1, p2 in bond_pairs:
        fig.add_trace(go.Scatter(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]],
            mode="lines", line=dict(color="white", width=2),
            showlegend=False,
        ))

    # Atoms
    if atom_screen:
        fig.add_trace(go.Scatter(
            x=list(atom_screen["x"]), y=list(atom_screen["y"]),
            mode="markers+text",
            marker=dict(
                size=14,
                color=[_ELEM_COLOURS.get(s, "#888888") for s in atom_screen["symbols"]],
                line=dict(color="black", width=1),
            ),
            text=atom_screen["symbols"],
            textposition="top center",
            textfont=dict(color="white", size=10),
            name="Atoms",
        ))

    _themes = {
        "dark":  dict(plot_bgcolor="#111111", paper_bgcolor="#1a1a1a", font_color="white"),
        "light": dict(plot_bgcolor="#ffffff", paper_bgcolor="#f5f5f5", font_color="black"),
        "navy":  dict(plot_bgcolor="#0a1628", paper_bgcolor="#0d1f3c", font_color="white"),
    }
    _t = _themes["dark"]
    fig.update_layout(
        title=f"Current density — slice at {axis_label} = {slice_pos:.2f}",
        xaxis_title=f"{ia_label} (Å)",
        yaxis_title=f"{ib_label} (Å)",
        yaxis_scaleanchor="x" if lock_aspect else None,
        plot_bgcolor=_t["plot_bgcolor"],
        paper_bgcolor=_t["paper_bgcolor"],
        font=dict(color=_t["font_color"]),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=plot_height,
        margin=dict(l=60, r=80, t=60, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# UI — sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Step 1 — Upload files")
    cur_file = st.file_uploader(
        "Current file (`current.dat` or `NMRCURBX/Y/Z`)",
        type=None,
        help="CASTEP binary or VASP plain-text format auto-detected.",
    )
    struct_file = st.file_uploader(
        "Structure file (`.cell` or `POSCAR`/`CONTCAR`)",
        type=None,
    )

    st.divider()
    st.header("Step 2 — Slice controls")

    axis_labels = {"a (first lattice vector)": 0,
                   "b (second lattice vector)": 1,
                   "c (third lattice vector)": 2}
    axis_label = st.radio("Slice axis", list(axis_labels.keys()))
    axis_idx = axis_labels[axis_label]

    slice_pos = st.slider("Slice position (fractional)", 0.0, 1.0, 0.5, 0.01)

    b_opts = {"Bx": 0, "By": 1, "Bz": 2}
    b_label = st.radio(
        "B-field direction (CASTEP only)",
        list(b_opts.keys()),
        help="Ignored for VASP (only one field direction stored).",
    )
    b_index = b_opts[b_label]

    step = st.slider("Arrow subsample (every Nth point)", 1, 10, 3)
    show_atoms = st.toggle("Show atoms", value=True)
    show_bonds = st.toggle("Show bonds", value=True)

    st.divider()
    with st.expander("Advanced arrow settings"):
        arrow_style = st.radio(
            "Arrow style",
            options=["lines", "quiver", "magnitude"],
            format_func={
                "lines": "Lines (simple segments)",
                "quiver": "Quiver (zoom-stable arrowheads)",
                "magnitude": "Magnitude-coloured lines",
            }.get,
            help=(
                "**Lines**: fast, uniform colour.  "
                "**Quiver**: `ff.create_quiver` arrowheads — scale with the data axes on zoom.  "
                "**Magnitude**: arrows coloured by |j|."
            ),
        )
        normalize = st.toggle(
            "Normalise arrow length",
            value=True,
            help="On: all arrows same length (direction only). Off: length ∝ |j|.",
        )
        arrow_scale = st.slider(
            "Arrow scale", 0.05, 1.0, 0.3, 0.05,
            help="Overall arrow length multiplier.",
        )
        arrow_width = st.slider("Arrow line width (px)", 1, 4, 1)

        if arrow_style == "quiver":
            arrow_head_scale = st.slider(
                "Arrowhead size", 0.05, 0.8, 0.3, 0.05,
                help="Relative size of the V-shaped arrowhead.",
            )
        else:
            arrow_head_scale = 0.3

        if arrow_style == "magnitude":
            arrow_colorscale = st.selectbox(
                "Arrow colorscale",
                ["Plasma", "Inferno", "Magma", "Turbo", "RdBu", "Viridis"],
                index=0,
            )
            arrow_color = "white"  # unused
        else:
            arrow_color = st.color_picker("Arrow colour", "#ffffff")
            arrow_colorscale = "Plasma"  # unused

    with st.expander("Advanced plot settings"):
        heatmap_colorscale = st.selectbox(
            "Heatmap colorscale",
            ["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "RdYlBu",
             "Turbo", "Hot", "Greys", "Electric"],
            index=0,
        )
        heatmap_opacity = st.slider("Heatmap opacity", 0.1, 1.0, 0.5, 0.05)
        heatmap_clim_mode = st.radio(
            "Colour limits",
            options=["auto", "fraction", "absolute"],
            format_func={"auto": "Auto", "fraction": "Fraction of max", "absolute": "Absolute"}.get,
            horizontal=True,
            help=(
                "**Auto**: full data range.  "
                "**Fraction**: set limits as fractions of the slice maximum (0–1).  "
                "**Absolute**: set limits in |j| units."
            ),
        )
        if heatmap_clim_mode == "fraction":
            _clim_cols = st.columns(2)
            heatmap_clim_lo = _clim_cols[0].number_input("Low", 0.0, 1.0, 0.0, 0.01, format="%.2f")
            heatmap_clim_hi = _clim_cols[1].number_input("High", 0.0, 1.0, 1.0, 0.01, format="%.2f")
        elif heatmap_clim_mode == "absolute":
            _clim_cols = st.columns(2)
            heatmap_clim_lo = _clim_cols[0].number_input("Low", value=0.0, format="%.4f")
            heatmap_clim_hi = _clim_cols[1].number_input("High", value=1.0, format="%.4f")
        else:
            heatmap_clim_lo, heatmap_clim_hi = 0.0, 1.0  # unused
        lock_aspect = st.toggle(
            "Lock aspect ratio (equal axes)",
            value=True,
            help="Keep x and y axes the same scale. Uncheck to fill the panel.",
        )
        plot_height = st.slider("Plot height (px)", 300, 1400, 700, 50)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if cur_file is None or struct_file is None:
    st.info("Upload both files in the sidebar to begin.")
    st.stop()

cur_bytes = cur_file.read()
struct_bytes = struct_file.read()

try:
    cd, atoms, fmt = load_current_data(cur_bytes, struct_bytes)
except Exception as exc:
    st.error(f"Failed to parse files: {exc}")
    st.stop()

# Info bar
nx, ny, nz = cd.grid
lat = cd.lattice
st.markdown(
    f"**Format:** `{fmt}` &nbsp;|&nbsp; "
    f"**Grid:** {nx}×{ny}×{nz} &nbsp;|&nbsp; "
    f"**Cell diagonals:** {lat[0,0]:.3f} × {lat[1,1]:.3f} × {lat[2,2]:.3f} Å"
)

# Compute slice and render
try:
    scr_x, scr_y, j_u1, j_u2, j_mag, atom_screen, bond_pairs = compute_slice(
        cd, atoms, axis_idx, slice_pos, b_index, step, show_atoms, show_bonds,
    )
    _all_labels = ["a", "b", "c"]
    _ia_label = _all_labels[[i for i in range(3) if i != axis_idx][0]]
    _ib_label = _all_labels[[i for i in range(3) if i != axis_idx][1]]
    fig = build_figure(
        scr_x, scr_y, j_u1, j_u2, j_mag, atom_screen, bond_pairs,
        axis_label=["a", "b", "c"][axis_idx],
        slice_pos=slice_pos,
        ia_label=_ia_label,
        ib_label=_ib_label,
        arrow_style=arrow_style,
        arrow_scale=arrow_scale,
        arrow_color=arrow_color,
        arrow_width=arrow_width,
        normalize=normalize,
        arrow_head_scale=arrow_head_scale,
        arrow_colorscale=arrow_colorscale,
        heatmap_colorscale=heatmap_colorscale,
        heatmap_opacity=heatmap_opacity,
        heatmap_clim_mode=heatmap_clim_mode,
        heatmap_clim_lo=heatmap_clim_lo,
        heatmap_clim_hi=heatmap_clim_hi,
        lock_aspect=lock_aspect,
        plot_height=plot_height,
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as exc:
    st.error(f"Slice computation failed: {exc}")
