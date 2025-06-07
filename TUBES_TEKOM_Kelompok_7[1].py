import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Untuk plot 3D
import matplotlib.animation as animation # Untuk animasi
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import newton

# --- Tahap 1: Definisi Konstanta & Kondisi Awal ---
print("Tahap 1: Definisi Konstanta & Kondisi Awal")

G = 6.674e-11  # Nm^2/kg^2
M_BUMI = 5.972e24  # kg
M_BULAN = 7.348e22  # kg
R_BUMI = 6.371e6    # meter (Radius Bumi)
R_BULAN = 1.737e6   # meter (Radius Bulan)

# Kondisi Awal Bulan (3D)
r_awal_x = 10 * R_BUMI  # Posisi awal sumbu x (3.844e8 untuk simulasi normal)
r_awal_y = 0           # Posisi awal sumbu y
r_awal_z = 2 * R_BUMI  # Posisi awal sumbu z (agar ada variasi di z) (0 untuk simulasi normal)

v_awal_x = 0           # Kecepatan awal x
v_awal_y = 1000       # Kecepatan awal y (tangensial) (1022 untuk simulasi normal)
v_awal_z = -200        # Kecepatan awal z (untuk membuat geraknya menjadi 3D) (0 untuk simulasi normal)

# State vector awal: S = [x, y, z, vx, vy, vz]
S0 = [r_awal_x, r_awal_y, r_awal_z, v_awal_x, v_awal_y, v_awal_z]

# Waktu simulasi
t_awal = 0
t_akhir = 3600 * 24 * 3 # (Bisa disesuaikan sesuai kebutuhan)
dt = 180 # Delta t dalam detik (Bisa disesuaikan untuk titik di animasi)
t_eval = np.arange(t_awal, t_akhir, dt)

print(f"Kondisi awal Bulan (3D): Posisi ({S0[0]/1000:.0f} km, {S0[1]/1000:.0f} km, {S0[2]/1000:.0f} km), Kec. ({S0[3]:.0f} m/s, {S0[4]:.0f} m/s, {S0[5]:.0f} m/s)")

# --- Tahap 2: Model Matematis & Fungsi PDB ---
print("\nTahap 2: Model Matematis (Fungsi PDB 3D)")

def sistem_pdb_3d(t, S):
    x, y, z, vx, vy, vz = S
    # Jarak dari pusat Bumi ke Bulan
    jarak_satelit_ke_pusat = np.sqrt(x**2 + y**2 + z**2)
    if jarak_satelit_ke_pusat == 0: 
        return [0,0,0,0,0,0]
        
    r_kubik = jarak_satelit_ke_pusat**3
    
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -G * M_BUMI * x / r_kubik
    dvydt = -G * M_BUMI * y / r_kubik
    dvzdt = -G * M_BUMI * z / r_kubik
    
    return [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt]

# --- Tahap 3: Penerapan Metode Numerik ---
print("\nTahap 3.a: Menyelesaikan PDB dengan Runge-Kutta (solve_ivp)")
sol = solve_ivp(sistem_pdb_3d, [t_awal, t_akhir], S0, t_eval=t_eval, method='RK45',  dense_output=True ) #tambahkan rtol=1e-9, atol=1e-12 untuk simulasi normal agar tidak terjadir ERROR dalam simulasi

t = sol.t
x_bulan = sol.y[0,:]
y_bulan = sol.y[1,:]
z_bulan = sol.y[2,:]
# vx_bulan = sol.y[3,:] 
# vy_bulan = sol.y[4,:]
# vz_bulan = sol.y[5,:]

print(f"Simulasi PDB selesai. Jumlah titik data: {len(t)}")

# Metode 2: Interpolasi
print("\nTahap 3.b: Interpolasi Hasil PDB (untuk Jarak)")
jarak_tumbukan_ideal = R_BUMI + R_BULAN
t_tumbukan_nr = None # DiInisialisasi

if len(t) > 3 : # Ini untuk menghitung beberapa poin untuk interpolasi yang baik
    jarak_bulan_bumi_3d = np.sqrt(x_bulan**2 + y_bulan**2 + z_bulan**2)
    
    unique_t, unique_indices = np.unique(t, return_index=True)
    if len(unique_t) > 3: # untuk lebih aman > k (k=3) kami pilih karena cubicSpline butuh minimal 4 poin jika bc_type tidak diset
        interp_jarak_3d = CubicSpline(unique_t, jarak_bulan_bumi_3d[unique_indices])
        print("Interpolasi jarak (3D) berhasil dibuat.")

        # Metode 3: Newton-Raphson
        print("\nTahap 3.c: Mencari Waktu Tumbukan dengan Newton-Raphson (3D)")
        def fungsi_tumbukan_3d(waktu_cek):
            if waktu_cek < unique_t[0] or waktu_cek > unique_t[-1]:
                return 1e12 
            return interp_jarak_3d(waktu_cek) - jarak_tumbukan_ideal

        indeks_mendekati_tumbukan = np.where(jarak_bulan_bumi_3d <= jarak_tumbukan_ideal)[0]
        
        if len(indeks_mendekati_tumbukan) > 0:
            t_tebakan_awal = t[indeks_mendekati_tumbukan[0]]
            try:
                # Memastikan tebakan awal ada dalam rentang interpolasi yang sesuai
                if t_tebakan_awal < unique_t[0]: t_tebakan_awal = unique_t[0]
                if t_tebakan_awal > unique_t[-1]: t_tebakan_awal = unique_t[-1]

                t_tumbukan_nr = newton(fungsi_tumbukan_3d, x0=t_tebakan_awal)
                print(f"Estimasi waktu tumbukan (NR): {t_tumbukan_nr/3600:.2f} jam")
                # Menempatkan posisi xyz saat tumbukan dari solusi dense output solve_ivp
                pos_tumbukan_xyz = sol.sol(t_tumbukan_nr)[:3]
                jarak_saat_nr = np.linalg.norm(pos_tumbukan_xyz)
                print(f"Jarak pada waktu tsb (interpolasi PDB): {jarak_saat_nr/1000:.2f} km")
                print(f"Target jarak tumbukan: {jarak_tumbukan_ideal/1000:.2f} km")

            except RuntimeError as e:
                print(f"Newton-Raphson gagal konvergen: {e}. Coba tebakan awal atau parameter lain.")
            except ValueError as e:
                print(f"Error pada Newton-Raphson: {e}.")
        else:
            print("Bulan tidak menumbuk Bumi dalam rentang waktu simulasi.")
            idx_min_jarak = np.argmin(jarak_bulan_bumi_3d)
            print(f"Jarak terdekat: {jarak_bulan_bumi_3d[idx_min_jarak]/1000:.2f} km (pada t={t[idx_min_jarak]/3600:.2f} jam)")
            print(f"Target jarak tumbukan: {jarak_tumbukan_ideal/1000:.2f} km")
    else:
        print("Data waktu tidak valid/cukup untuk CubicSpline. Lewati interpolasi & Newton-Raphson.")
else:
    print("Tidak cukup data untuk interpolasi dan Newton-Raphson.")


# --- Tahap 4: Visualisasi Hasil Perhitungan ---
print("\nTahap 4: Visualisasi Hasil (3D)")

# --- Plot Statis 3D ---
fig_static_3d = plt.figure(figsize=(10, 8))
ax_static_3d = fig_static_3d.add_subplot(111, projection='3d')

# Plot Bumi sebagai bola (permukaan)
u_earth = np.linspace(0, 2 * np.pi, 100)
v_earth = np.linspace(0, np.pi, 100)
x_earth_surf = R_BUMI * np.outer(np.cos(u_earth), np.sin(v_earth))
y_earth_surf = R_BUMI * np.outer(np.sin(u_earth), np.sin(v_earth))
z_earth_surf = R_BUMI * np.outer(np.ones(np.size(u_earth)), np.cos(v_earth))
ax_static_3d.plot_surface(x_earth_surf, y_earth_surf, z_earth_surf, color='deepskyblue', alpha=0.6, label='Bumi')

# Plot lintasan Bulan
ax_static_3d.plot(x_bulan, y_bulan, z_bulan, label='Lintasan Bulan (RK4)', color='grey')

# Plot titik tumbukan jika ditemukan dan valid
if t_tumbukan_nr is not None and t_tumbukan_nr >= t_awal and t_tumbukan_nr <= t_akhir:
    try:
        pos_tumbukan_xyz_plot = sol.sol(t_tumbukan_nr)[:3] # Ambil x,y,z
        ax_static_3d.scatter(pos_tumbukan_xyz_plot[0], pos_tumbukan_xyz_plot[1], pos_tumbukan_xyz_plot[2], 
                             color='red', s=100, label=f'Tumbukan (NR) t={t_tumbukan_nr/3600:.2f} jam', depthshade=False)
    except Exception as e:
        print(f"Tidak bisa plot titik tumbukan: {e}")


# Pengaturan Axes untuk plot statis 3D
max_range = np.max(np.abs(np.concatenate([x_bulan, y_bulan, z_bulan]))) * 1.1
ax_static_3d.set_xlim([-max_range, max_range])
ax_static_3d.set_ylim([-max_range, max_range])
ax_static_3d.set_zlim([-max_range, max_range])
ax_static_3d.set_xlabel("Posisi X (meter)")
ax_static_3d.set_ylabel("Posisi Y (meter)")
ax_static_3d.set_zlabel("Posisi Z (meter)")
ax_static_3d.set_title("Simulasi Jatuhnya Bulan ke Bumi (3D Statis)")
# Untuk membuat pola sedikit tricky untuk plot_surface, kita tambahkan proxy artist
# bumi_proxy = plt.Rectangle((0, 0), 1, 1, fc="deepskyblue", alpha=0.6)
# ax_static_3d.legend([bumi_proxy, ax_static_3d.get_lines()[0]], ['Bumi', 'Lintasan Bulan'])
# Atau cara lebih simpel:
# ax_static_3d.legend() # Ini mungkin tidak menampilkan label surface dengan baik
print("Plot statis 3D ditampilkan. Tutup untuk melanjutkan ke plot jarak & animasi.")
plt.show()


# --- Plot Jarak vs Waktu ---
if 'jarak_bulan_bumi_3d' in locals() and len(t) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(t / 3600, jarak_bulan_bumi_3d / 1000, label='Jarak Bulan-Bumi (RK4)')
    if 'interp_jarak_3d' in locals() and callable(interp_jarak_3d) and len(unique_t) > 1:
        t_halus = np.linspace(unique_t[0], unique_t[-1], 500)
        plt.plot(t_halus / 3600, interp_jarak_3d(t_halus) / 1000, '--', label='Jarak (Interpolasi Spline)', alpha=0.7)
    
    plt.axhline(jarak_tumbukan_ideal / 1000, color='red', linestyle=':', label='Jarak Tumbukan Ideal')
    if t_tumbukan_nr is not None and t_tumbukan_nr >= t_awal and t_tumbukan_nr <= t_akhir:
         plt.plot(t_tumbukan_nr / 3600, interp_jarak_3d(t_tumbukan_nr) / 1000, 'ro', label='Titik Tumbukan (NR)')

    plt.xlabel("Waktu (jam)")
    plt.ylabel("Jarak Pusat Bumi - Pusat Bulan (km)")
    plt.title("Jarak Bulan ke Bumi Seiring Waktu (3D)")
    plt.legend()
    plt.grid(True)
    print("Plot jarak vs waktu akan ditampilkan. Tutup untuk melanjutkan ke animasi 3D.")
    plt.show()
else:
    print("Tidak ada data jarak untuk diplot.")


# --- Animasi 3D ---
print("\nMenyiapkan animasi 3D...")
fig_anim_3d = plt.figure(figsize=(10, 8))
ax_anim_3d = fig_anim_3d.add_subplot(111, projection='3d')

# Plot Bumi (statis dalam animasi)
ax_anim_3d.plot_surface(x_earth_surf, y_earth_surf, z_earth_surf, color='deepskyblue', alpha=0.3, zorder=1)

# Untuk Garis lintasan dan titik Bulan yang akan dianimasikan
line_anim, = ax_anim_3d.plot([], [], [], lw=1, color='grey', label='Lintasan Bulan', zorder=2) # Lintasan yang sudah dilalui
point_anim, = ax_anim_3d.plot([], [], [], 'o', color='red', markersize=5, label='Bulan', zorder=3) # Posisi Bulan saat ini

# Pengaturan Axes untuk animasi
ax_anim_3d.set_xlim([-max_range, max_range])
ax_anim_3d.set_ylim([-max_range, max_range])
ax_anim_3d.set_zlim([-max_range, max_range])
ax_anim_3d.set_xlabel("X (m)")
ax_anim_3d.set_ylabel("Y (m)")
ax_anim_3d.set_zlabel("Z (m)")
ax_anim_3d.set_title("Animasi Jatuhnya Bulan ke Bumi (3D)")
ax_anim_3d.legend(loc='upper right')

# Teks untuk menampilkan waktu simulasi
time_text_anim = ax_anim_3d.text2D(0.05, 0.95, '', transform=ax_anim_3d.transAxes)

# Jumlah frame dan skip (untuk mempercepat animasi jika data terlalu banyak)
num_frames_total = len(t)
# Atur frame_skip, misal jika num_frames_total > 300, skip beberapa
frame_skip = 1
if num_frames_total > 300:
    frame_skip = int(num_frames_total / 300)
    if frame_skip == 0: frame_skip = 1 # Pastikan tidak nol
effective_num_frames = num_frames_total // frame_skip


def init_anim_3d():
    line_anim.set_data_3d([], [], [])
    point_anim.set_data_3d([], [], [])
    time_text_anim.set_text('')
    return line_anim, point_anim, time_text_anim

def animate_3d(i):
    actual_index = i * frame_skip
    # Update lintasan yang sudah dilalui
    line_anim.set_data_3d(x_bulan[:actual_index+1], y_bulan[:actual_index+1], z_bulan[:actual_index+1])
    # Update posisi Bulan saat ini
    point_anim.set_data_3d([x_bulan[actual_index]], [y_bulan[actual_index]], [z_bulan[actual_index]])
    
    time_text_anim.set_text(f'Waktu: {t[actual_index]/3600:.2f} jam')
    
    # Jika Bulan sudah menumbuk (berdasarkan Newton-Raphson) dan frame saat ini melewati waktu tumbukan
    if t_tumbukan_nr is not None and t[actual_index] >= t_tumbukan_nr:
        # Tandai titik tumbukan secara visual di animasi (jika belum ditandai)
        # Kode Ini bisa di-edit untuk mengubah warna Bulan atau menghentikan animasi
        point_anim.set_color('black')
        point_anim.set_markersize(10)
        time_text_anim.set_text(f'TUMBUKAN! Waktu: {t_tumbukan_nr/3600:.2f} jam (NR)')
        # Hentikan animasi setelah tumbukan dengan mengembalikan list kosong
        # Setelah bulan menumbuk,animasi tetap berjalan tapi Bulan jadi hitam.

    return line_anim, point_anim, time_text_anim

# Membuat animasi
# blit=True bisa mempercepat, untuk mempercepat animasi tapi jika error, gunakan blit=False.
# interval merupakan delay antar frame dalam milidetik
ani_3d = animation.FuncAnimation(fig_anim_3d, animate_3d, frames=effective_num_frames,
                                 init_func=init_anim_3d, blit=False, interval=50) # blit=False lebih aman untuk 3D

print("Animasi 3D akan ditampilkan. Ini mungkin butuh beberapa saat untuk render.")
plt.show()

print("\n--- Selesai ---")
print("Lanjut kerjakan laporan !")
