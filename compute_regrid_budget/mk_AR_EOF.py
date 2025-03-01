import numpy as np
import xarray as xr

from sklearn.decomposition import PCA

import argparse

parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--input', type=str, help='Input file', required=True)
parser.add_argument('--output', type=str, help='Output file', required=True)
parser.add_argument('--nPCA', type=int, help='Number of components', default=5)
args = parser.parse_args()
print(args)

ds = xr.open_dataset(args.input)


#ds = ds.where(ds.coords['lat'] > 20)

print(list(ds.keys()))

# Need to remove the mean
print("Remove the climatology")
count_mean = ds["count"].mean(dim="time")
count_anom = ds["count"] - count_mean

# making_mapping

idx = np.isfinite(count_anom[0, :, :].to_numpy())
numbering = np.array(list(range(count_mean.size)), dtype=int).reshape(len(ds.coords["lat"]), len(ds.coords["lon"]))
extracted_numbering = numbering[idx] 

data = np.zeros((len(ds.coords["time"]), np.sum(idx), ))

for i in range(len(ds.coords["time"])):
    data[i, :] = count_anom.isel(time=i).to_numpy()[idx]

print(np.sum(np.isnan(data)))


print("Computing EOF...")
pca = PCA(n_components=args.nPCA)

pca.fit(data)

for i in range(pca.n_components_):
    print("Length of the %d-th component: %f" % (i, np.sum(pca.components_[i, :]**2)**0.5))


print("Outputting EOF...")
        
EOF_count = np.zeros((pca.n_components_, len(ds.coords["lat"]) * len(ds.coords["lon"]),))
EOF_count[:, :] = np.nan

for i in range(pca.n_components_):
    for j, target_j in enumerate(extracted_numbering):
        EOF_count[i, target_j] = pca.components_[i, j]

EOF_count = EOF_count.reshape((pca.n_components_, len(ds.coords["lat"]), len(ds.coords["lon"]),))


EOF_amps = np.zeros((pca.n_components_, len(ds.coords["time"])))

for i in range(EOF_amps.shape[0]):
    for j in range(EOF_amps.shape[1]):
        EOF_amps[i, j] = np.sum(pca.components_[i, :] * data[j, :])


    
EOF_amps_normalized = EOF_amps.copy()

for i in range(EOF_amps_normalized.shape[0]):
    EOF_amps_normalized[i, :] /= EOF_amps[i, :].std()

EOF_ds = xr.Dataset(

    data_vars = dict(
        count_EOF = (["EOF", "lat", "lon"], EOF_count),
        amps = (["EOF", "time"], EOF_amps),
        amps_normalized = (["EOF", "time"], EOF_amps_normalized),
        explained_variance = (["EOF",], pca.explained_variance_),
        explained_variance_ratio = (["EOF",], pca.explained_variance_ratio_),
    ),

    coords = dict(
        EOF        = list(range(pca.n_components_)),
        lat        = ds.coords["lat"],
        lon        = ds.coords["lon"],
        time       = ds.coords["time"],
    ),

)

for i in range(len(EOF_ds.coords["EOF"])):
    
    sign = 1 if (EOF_ds["count_EOF"].isel(EOF=i).sel(lat=35.0, lon=235.0, method="nearest").to_numpy() > 0) else -1
    EOF_ds["count_EOF"][i, :, :] *= sign
    EOF_ds["amps"][i, :] *= sign
    EOF_ds["amps_normalized"][i, :] *= sign





EOF_ds.to_netcdf(args.output)
