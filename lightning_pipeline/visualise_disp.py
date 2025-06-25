import numpy as np
import plotly.graph_objects as go
from skimage import measure
import nibabel as nib
import plotly.express as px

def plot_disp(ddf, mask, save_path = None):
    step = 10 # Sampling step to reduce the number of vectors for clarity
    x, y, z = np.where(mask)

    # Subsample the vectors
    x = x[::step]
    y = y[::step]
    z = z[::step]

    vectors = ddf[x, y, z]
    magnitudes = np.linalg.norm(vectors, axis=1)

    # Apply threshold: only keep vectors with magnitude ≤ threshold
    valid_indices = magnitudes <= 10
    x, y, z = x[valid_indices], y[valid_indices], z[valid_indices]
    vectors = vectors[valid_indices]
    magnitudes = magnitudes[valid_indices]

    # Extract displacement components
    u, v, w = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    # Normalize magnitudes for color scaling
    norm_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())

    # Choose a colorscale
    colorscale = 'Viridis'  # You can choose other Plotly colorscales

    # Generate colors based on magnitudes using sample_colorscale
    colors = px.colors.sample_colorscale(colorscale, norm_magnitudes)

    # Prepare lines with colors and transparency
    x_lines, y_lines, z_lines, line_colors = [], [], [], []
    end_x, end_y, end_z = x + u, y + v, z + w
    for i in range(len(x)):
        x_lines.extend([x[i], end_x[i], None])
        y_lines.extend([y[i], end_y[i], None])
        z_lines.extend([z[i], end_z[i], None])
        # Repeat color for each segment (start and end points, None)
        rgba_color = colors[i].replace('rgb', 'rgba').replace(')', ', 0.8)')
        line_colors.extend([rgba_color, rgba_color, 'rgba(0,0,0,0)'])  # Transparent for 'None'

    disp = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode='lines',
        line=dict(color=line_colors, width=2),
        hoverinfo='text',
        hovertemplate='<b>Displacement</b><br>Magnitude: %{text:.2f}<extra></extra>',
        text=np.repeat(magnitudes, 3),
        name='Displacements'
    )

    # Optionally, add a mask surface for better context
    verts, faces, normals, values = measure.marching_cubes(mask.astype(float), level=0.5)
    mask_surface = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.1,
        color='lightgrey',
        name='Mask Surface',
        showscale=False
    )

    # Create color bar for magnitudes
    color_bar = go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        marker=dict(
            colorscale=colorscale,
            cmin=magnitudes.min(),
            cmax=magnitudes.max(),
            color=magnitudes,
            opacity=0.8,
            colorbar=dict(
                title='Displacement Magnitude',
                thickness=10,
                len=0.7
            )
        ),
        showlegend=False
    )

    fig = go.Figure(data=[mask_surface, disp, color_bar])

    fig.update_layout(
        scene=dict(
            aspectmode='data',  # Ensures the aspect ratio is based on the data
            xaxis=dict(title_text='X', backgroundcolor='rgb(230, 230,230)', gridcolor='white', showbackground=True, zerolinecolor='white'),
            yaxis=dict(title_text='Y', backgroundcolor='rgb(230, 230,230)', gridcolor='white', showbackground=True, zerolinecolor='white'),
            zaxis=dict(title_text='Z', backgroundcolor='rgb(230, 230,230)', gridcolor='white', showbackground=True, zerolinecolor='white'),
        ),
        title='3D Displacement Vectors with Magnitude-Based Coloring',
        legend=dict(
            itemsizing='constant'
        ),
        width=800,
        height=800,
        template='plotly_white'
    )

    fig.write_html(save_path) if save_path else fig.show()

# if __name__ == "__main__":
#     ddf = np.load('/gpfs/space/projects/BetterMedicine/illia/bm-ai-pipelines/data/datasets/TUH/nifti_subset_reorient_false/0d26c5c3-c652-17fd-7a35-b25b4a3d6cb9/displacement.npy')[0]
#     mask = nib.load('/gpfs/space/projects/BetterMedicine/illia/bm-ai-pipelines/data/datasets/TUH/nifti_subset_reorient_false/0d26c5c3-c652-17fd-7a35-b25b4a3d6cb9/baseline/2.25.40274311684839462555706445541819421933_lungmask_preprocessed_pad.nii.gz').get_fdata().astype(bool)
#     plot_disp(ddf, mask, 'disp.html')
