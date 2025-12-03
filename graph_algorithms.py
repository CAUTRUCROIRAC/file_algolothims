import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use('Agg')

class GraphAlgorithmsVisualizer:
    def __init__(self):
        self.create_sample_graphs()
    
    def create_sample_graphs(self):
        """Tạo đồ thị mẫu cho các thuật toán"""
        # Đồ thị cho Prim và Kruskal (vô hướng có trọng số)
        self.graph_weighted = nx.Graph()
        edges_weighted = [
            (0, 1, 4), (0, 7, 8), (1, 2, 8), (1, 7, 11),
            (2, 3, 7), (2, 5, 4), (2, 8, 2), (3, 4, 9),
            (3, 5, 14), (4, 5, 10), (5, 6, 2), (6, 7, 1),
            (6, 8, 6), (7, 8, 7)
        ]
        self.graph_weighted.add_weighted_edges_from(edges_weighted)
        
        # Đồ thị có hướng cho Ford-Fulkerson
        self.graph_directed = nx.DiGraph()
        edges_directed = [
            ('s', 'a', 16), ('s', 'c', 13), ('a', 'c', 10),
            ('a', 'b', 12), ('c', 'a', 4), ('c', 'd', 14),
            ('b', 'c', 9), ('b', 't', 20), ('d', 'b', 7),
            ('d', 't', 4)
        ]
        for u, v, w in edges_directed:
            self.graph_directed.add_edge(u, v, capacity=w)
        
        # Đồ thị Euler cho Fleury và Hierholzer
        self.graph_euler = nx.Graph()
        edges_euler = [
            (0, 1), (0, 2), (1, 2), (2, 3), (3, 4),
            (4, 5), (5, 2), (3, 6), (6, 7), (7, 3)
        ]
        self.graph_euler.add_edges_from(edges_euler)
    
    def visualize_prim(self):
        """Trực quan hóa thuật toán Prim"""
        st.subheader("📊 Thuật toán Prim - Cây khung nhỏ nhất")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Giới thiệu:**")
            st.write("""
            Thuật toán Prim tìm cây khung nhỏ nhất (MST) cho đồ thị vô hướng có trọng số.
            - Bắt đầu từ một đỉnh bất kỳ
            - Luôn thêm cạnh có trọng số nhỏ nhất nối cây với đỉnh chưa thuộc cây
            - Độ phức tạp: O(E log V)
            """)
            
            st.markdown("**Bước thực hiện:**")
            st.code("""
            1. Chọn đỉnh bắt đầu
            2. Khởi tạo cây chỉ chứa đỉnh đó
            3. Trong khi cây chưa chứa tất cả đỉnh:
               - Tìm cạnh nhỏ nhất nối đỉnh trong cây với đỉnh ngoài cây
               - Thêm cạnh và đỉnh đó vào cây
            4. Kết quả là cây khung nhỏ nhất
            """)
        
        # Thực hiện thuật toán Prim
        mst_prim = nx.minimum_spanning_tree(self.graph_weighted, algorithm='prim')
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Vẽ đồ thị gốc
            pos = nx.spring_layout(self.graph_weighted, seed=42)
            nx.draw(self.graph_weighted, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, ax=ax1, font_weight='bold')
            labels = nx.get_edge_attributes(self.graph_weighted, 'weight')
            nx.draw_networkx_edge_labels(self.graph_weighted, pos, edge_labels=labels, ax=ax1)
            ax1.set_title("Đồ thị gốc")
            
            # Vẽ cây khung nhỏ nhất từ Prim
            nx.draw(self.graph_weighted, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, ax=ax2, font_weight='bold')
            nx.draw_networkx_edges(self.graph_weighted, pos, edgelist=mst_prim.edges(), 
                                  edge_color='red', width=2, ax=ax2)
            nx.draw_networkx_edge_labels(self.graph_weighted, pos, edge_labels=labels, ax=ax2)
            ax2.set_title("Cây khung nhỏ nhất (Prim)")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Hiển thị thông tin
        st.markdown("**Thông tin cây khung:**")
        total_weight = sum(self.graph_weighted.edges[edge]['weight'] for edge in mst_prim.edges())
        st.write(f"- Tổng trọng số: **{total_weight}**")
        st.write(f"- Số cạnh: **{mst_prim.number_of_edges()}**")
        st.write(f"- Số đỉnh: **{mst_prim.number_of_nodes()}**")
    
    def visualize_kruskal(self):
        """Trực quan hóa thuật toán Kruskal"""
        st.subheader("📊 Thuật toán Kruskal - Cây khung nhỏ nhất")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Giới thiệu:**")
            st.write("""
            Thuật toán Kruskal tìm cây khung nhỏ nhất (MST) cho đồ thị vô hướng có trọng số.
            - Sắp xếp các cạnh theo trọng số tăng dần
            - Thêm cạnh vào cây nếu không tạo chu trình
            - Sử dụng cấu trúc Union-Find để kiểm tra chu trình
            - Độ phức tạp: O(E log E)
            """)
            
            st.markdown("**Bước thực hiện:**")
            st.code("""
            1. Sắp xếp tất cả cạnh theo trọng số tăng dần
            2. Khởi tạo rừng (mỗi đỉnh là một cây)
            3. Duyệt qua các cạnh đã sắp xếp:
               - Nếu cạnh nối 2 cây khác nhau, thêm vào MST
               - Hợp nhất 2 cây
            4. Dừng khi có đủ (V-1) cạnh
            """)
        
        # Thực hiện thuật toán Kruskal
        mst_kruskal = nx.minimum_spanning_tree(self.graph_weighted, algorithm='kruskal')
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            pos = nx.spring_layout(self.graph_weighted, seed=42)
            
            # Vẽ đồ thị gốc
            nx.draw(self.graph_weighted, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, ax=ax1, font_weight='bold')
            labels = nx.get_edge_attributes(self.graph_weighted, 'weight')
            nx.draw_networkx_edge_labels(self.graph_weighted, pos, edge_labels=labels, ax=ax1)
            ax1.set_title("Đồ thị gốc")
            
            # Vẽ cây khung nhỏ nhất từ Kruskal
            nx.draw(self.graph_weighted, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, ax=ax2, font_weight='bold')
            nx.draw_networkx_edges(self.graph_weighted, pos, edgelist=mst_kruskal.edges(), 
                                  edge_color='green', width=2, ax=ax2)
            nx.draw_networkx_edge_labels(self.graph_weighted, pos, edge_labels=labels, ax=ax2)
            ax2.set_title("Cây khung nhỏ nhất (Kruskal)")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # So sánh với Prim
        st.markdown("**So sánh Prim vs Kruskal:**")
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.info("**Prim**\n- Dựa trên đỉnh\n- Tốt cho đồ thị dày\n- Dùng heap")
        
        with col_comp2:
            st.info("**Kruskal**\n- Dựa trên cạnh\n- Tốt cho đồ thị thưa\n- Dùng Union-Find")
    
    def visualize_ford_fulkerson(self):
        """Trực quan hóa thuật toán Ford-Fulkerson"""
        st.subheader("📊 Thuật toán Ford-Fulkerson - Luồng cực đại")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Giới thiệu:**")
            st.write("""
            Thuật toán Ford-Fulkerson tìm luồng cực đại trong mạng luồng.
            - Tìm đường tăng luồng từ nguồn (s) đến đích (t)
            - Tăng luồng dọc theo đường tìm được
            - Lặp cho đến khi không còn đường tăng
            - Độ phức tạp: O(E * max_flow)
            """)
            
            st.markdown("**Các biến thể:**")
            st.write("- Edmonds-Karp (BFS tìm đường ngắn nhất)")
            st.write("- Dinic (sử dụng level graph)")
            
            st.markdown("**Bước thực hiện:**")
            st.code("""
            1. Khởi tạo luồng = 0
            2. Trong khi tồn tại đường từ s đến t:
               - Tìm đường tăng luồng (BFS/DFS)
               - Tìm giá trị tăng nhỏ nhất trên đường
               - Cập nhật luồng dọc theo đường
               - Cập nhật đồ thị dư
            3. Trả về luồng cực đại
            """)
        
        # Thực hiện thuật toán Ford-Fulkerson (Edmonds-Karp)
        flow_value, flow_dict = nx.maximum_flow(self.graph_directed, 's', 't')
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            pos = nx.spring_layout(self.graph_directed, seed=42)
            
            # Vẽ đồ thị gốc với capacities
            nx.draw(self.graph_directed, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, ax=ax1, font_weight='bold', arrowsize=20)
            edge_labels = nx.get_edge_attributes(self.graph_directed, 'capacity')
            nx.draw_networkx_edge_labels(self.graph_directed, pos, edge_labels=edge_labels, ax=ax1)
            ax1.set_title("Mạng luồng gốc (capacities)")
            
            # Vẽ luồng cực đại
            nx.draw(self.graph_directed, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, ax=ax2, font_weight='bold', arrowsize=20)
            
            # Tạo nhãn hiển thị luồng/capacity
            flow_edge_labels = {}
            for u in flow_dict:
                for v, flow in flow_dict[u].items():
                    if flow > 0:
                        flow_edge_labels[(u, v)] = f"{flow}/{edge_labels[(u, v)]}"
            
            nx.draw_networkx_edge_labels(self.graph_directed, pos, 
                                        edge_labels=flow_edge_labels, 
                                        font_color='red', ax=ax2)
            ax2.set_title(f"Luồng cực đại = {flow_value}")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Hiển thị thông tin luồng
        st.markdown(f"**Luồng cực đại tìm được: {flow_value}**")
        
        st.markdown("**Luồng trên từng cạnh:**")
        flow_data = []
        for u in sorted(flow_dict.keys()):
            for v in sorted(flow_dict[u].keys()):
                if flow_dict[u][v] > 0:
                    flow_data.append({
                        "Từ": u,
                        "Đến": v,
                        "Luồng": flow_dict[u][v],
                        "Capacity": edge_labels[(u, v)]
                    })
        
        st.table(pd.DataFrame(flow_data))
    
    def visualize_fleury(self):
        """Trực quan hóa thuật toán Fleury"""
        st.subheader("📊 Thuật toán Fleury - Tìm chu trình Euler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Giới thiệu:**")
            st.write("""
            Thuật toán Fleury tìm chu trình Euler trong đồ thị Euler.
            - Đồ thị Euler: mọi đỉnh có bậc chẵn
            - Bắt đầu từ đỉnh bất kỳ
            - Chọn cạnh không phải là cầu (nếu có thể)
            - Xóa cạnh đã đi qua
            - Độ phức tạp: O(E²)
            """)
            
            st.markdown("**Điều kiện Euler:**")
            st.write("- Đồ thị vô hướng: tất cả đỉnh bậc chẵn")
            st.write("- Đồ thị có hướng: bậc vào = bậc ra tại mọi đỉnh")
            
            st.markdown("**Bước thực hiện:**")
            st.code("""
            1. Kiểm tra điều kiện Euler
            2. Chọn đỉnh bắt đầu
            3. Trong khi còn cạnh:
               - Nếu có cạnh không phải cầu, chọn nó
               - Nếu chỉ còn cầu, chọn cầu
               - Thêm cạnh vào chu trình
               - Xóa cạnh khỏi đồ thị
            4. Trả về chu trình Euler
            """)
        
        # Kiểm tra đồ thị Euler
        is_eulerian = nx.is_eulerian(self.graph_euler)
        
        if is_eulerian:
            # Tìm chu trình Euler bằng networkx (dùng Hierholzer)
            euler_circuit = list(nx.eulerian_circuit(self.graph_euler))
        else:
            st.warning("⚠️ Đồ thị không phải đồ thị Euler!")
            return
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            pos = nx.spring_layout(self.graph_euler, seed=42)
            
            # Vẽ đồ thị gốc
            nx.draw(self.graph_euler, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, ax=ax1, font_weight='bold')
            ax1.set_title("Đồ thị Euler gốc")
            
            # Vẽ chu trình Euler
            nx.draw(self.graph_euler, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, ax=ax2, font_weight='bold')
            
            # Vẽ chu trình với màu sắc
            for i, (u, v) in enumerate(euler_circuit):
                nx.draw_networkx_edges(self.graph_euler, pos, edgelist=[(u, v)], 
                                      edge_color=f'C{i}', width=2, ax=ax2, 
                                      alpha=0.7)
            
            ax2.set_title("Chu trình Euler (Fleury)")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Hiển thị thông tin
        st.markdown("**Thông tin đồ thị:**")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.write("**Bậc các đỉnh:**")
            degrees = dict(self.graph_euler.degree())
            for node, deg in sorted(degrees.items()):
                st.write(f"- Đỉnh {node}: bậc {deg}")
        
        with col_info2:
            st.write("**Chu trình Euler:**")
            circuit_str = " → ".join([str(u) for u, _ in euler_circuit])
            circuit_str += f" → {euler_circuit[0][0]}"
            st.write(circuit_str)
            
            st.write(f"**Độ dài chu trình:** {len(euler_circuit)} cạnh")
    
    def visualize_hierholzer(self):
        """Trực quan hóa thuật toán Hierholzer"""
        st.subheader("📊 Thuật toán Hierholzer - Tìm chu trình Euler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Giới thiệu:**")
            st.write("""
            Thuật toán Hierholzer tìm chu trình Euler hiệu quả hơn Fleury.
            - Bắt đầu từ đỉnh bất kỳ
            - Tìm chu trình đơn giản
            - Hợp nhất các chu trình
            - Độ phức tạp: O(E)
            """)
            
            st.markdown("**Ưu điểm so với Fleury:**")
            st.write("- Không cần kiểm tra cầu")
            st.write("- Độ phức tạp tuyến tính")
            st.write("- Dễ cài đặt hơn")
            
            st.markdown("**Bước thực hiện:**")
            st.code("""
            1. Kiểm tra điều kiện Euler
            2. Chọn đỉnh bắt đầu, khởi tạo stack
            3. Trong khi stack không rỗng:
               - Lấy đỉnh u từ đỉnh stack
               - Nếu u còn cạnh chưa dùng:
                 - Chọn cạnh (u, v)
                 - Xóa cạnh, đẩy u và v vào stack
               - Ngược lại, thêm u vào chu trình
            4. Đảo ngược chu trình để có kết quả
            """)
        
        # Tìm chu trình Euler bằng Hierholzer
        is_eulerian = nx.is_eulerian(self.graph_euler)
        
        if is_eulerian:
            euler_circuit = list(nx.eulerian_circuit(self.graph_euler))
        else:
            st.warning("⚠️ Đồ thị không phải đồ thị Euler!")
            return
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            pos = nx.spring_layout(self.graph_euler, seed=42)
            
            # Vẽ đồ thị với chu trình Euler
            nx.draw(self.graph_euler, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, font_weight='bold', ax=ax)
            
            # Tô màu chu trình theo thứ tự
            for i, (u, v) in enumerate(euler_circuit):
                nx.draw_networkx_edges(self.graph_euler, pos, edgelist=[(u, v)], 
                                      edge_color=f'C{i}', width=3, ax=ax, 
                                      alpha=0.8, style='-')
            
            ax.set_title("Chu trình Euler (Hierholzer)")
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Hiển thị so sánh
        st.markdown("**So sánh Fleury vs Hierholzer:**")
        
        comp_data = {
            "Thuật toán": ["Fleury", "Hierholzer"],
            "Độ phức tạp": ["O(E²)", "O(E)"],
            "Kiểm tra cầu": ["Có", "Không"],
            "Khó cài đặt": ["Trung bình", "Dễ"],
            "Hiệu quả": ["Thấp", "Cao"]
        }
        
        st.table(pd.DataFrame(comp_data))
        
        # Minh họa từng bước
        st.markdown("**Minh họa từng bước Hierholzer:**")
        
        steps = [
            "1. Bắt đầu từ đỉnh 0",
            "2. Tìm chu trình đơn giản: 0-1-2-0",
            "3. Từ đỉnh 2, tìm chu trình: 2-3-4-5-2",
            "4. Từ đỉnh 3, tìm chu trình: 3-6-7-3",
            "5. Hợp nhất chu trình: 0-1-2-3-6-7-3-4-5-2-0"
        ]
        
        for step in steps:
            st.write(step)

def show_algorithms_info():
    """Hiển thị thông tin ứng dụng thuật toán trong giao thông"""
    st.markdown("---")
    st.markdown("""
    ### 📚 Lý thuyết ứng dụng trong giao thông
    
    **Cây khung nhỏ nhất (Prim/Kruskal):**
    - Thiết kế mạng lưới đường ít tốn kém nhất
    - Kết nối tất cả quận/huyện với chi phí tối thiểu
    - Quy hoạch hệ thống cáp viễn thông
    
    **Luồng cực đại (Ford-Fulkerson):**
    - Tối ưu hóa lưu lượng giao thông
    - Quản lý capacity tại các nút giao thông
    - Phân tích điểm tắc nghẽn trong thành phố
    
    **Chu trình Euler (Fleury/Hierholzer):**
    - Lập lộ trình thu gom rác tối ưu
    - Thiết kế tuyến xe buýt qua tất cả điểm
    - Kiểm tra đường đi của nhân viên giao hàng
    """)