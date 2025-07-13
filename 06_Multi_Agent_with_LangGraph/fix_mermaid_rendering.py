"""
Quick fix for the Mermaid rendering error in the notebook.
Replace the problematic cell with this code.
"""

# Replace the problematic cell with this:
import nest_asyncio
from IPython.display import Image, display, HTML
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

nest_asyncio.apply()

# Try multiple rendering methods
def safe_display_graph(compiled_graph, title="Graph"):
    """Safely display graph with multiple fallback methods"""
    
    # Method 1: Try Chrome instead of Pyppeteer
    try:
        display(
            Image(
                compiled_graph.get_graph().draw_mermaid_png(
                    curve_style=CurveStyle.LINEAR,
                    node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
                    wrap_label_n_words=9,
                    output_file_path=None,
                    draw_method=MermaidDrawMethod.CHROME,  # Changed from PYPPETEER
                    background_color="white",
                    padding=10,
                )
            )
        )
        print(f"✅ {title} rendered successfully!")
        return
    except Exception as e:
        print(f"❌ Chrome rendering failed: {e}")
    
    # Method 2: Show Mermaid syntax
    try:
        mermaid_syntax = compiled_graph.get_graph().draw_mermaid(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
        )
        
        print(f"\n📊 {title} - Mermaid Syntax:")
        print("=" * 50)
        print(mermaid_syntax)
        print("=" * 50)
        
        # Display as HTML
        html_content = f"""
        <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4>{title}</h4>
            <pre style="background: white; padding: 10px; border-radius: 3px; overflow-x: auto;">{mermaid_syntax}</pre>
        </div>
        """
        display(HTML(html_content))
        return
    except Exception as e:
        print(f"❌ Mermaid syntax failed: {e}")
    
    # Method 3: Simple text representation
    try:
        graph = compiled_graph.get_graph()
        print(f"\n📋 {title} - Graph Structure:")
        print("=" * 50)
        print("🏗️  Nodes:", list(graph.nodes.keys()))
        print("🔗 Edges:", [(edge.source, edge.target) for edge in graph.edges])
        print("🎯 Entry Point:", graph.entry_point)
        print("=" * 50)
    except Exception as e:
        print(f"❌ Text representation failed: {e}")

# Use this instead of the original display code:
safe_display_graph(compiled_research_graph, "Research Team Graph") 