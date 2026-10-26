package vn.tvd.detect;

import java.util.logging.Logger;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * ONNX Runtime session options: CUDA when available (set TVD_DEVICE=cpu to force CPU).
 * CUDA needs libcudart/libcublas 13 and libcudnn 9 on LD_LIBRARY_PATH; java/tvd.sh sets that up.
 */
public final class Onnx {
    private static final Logger LOG = Logger.getLogger("tvd");
    private static Boolean cuda;

    private Onnx() {
    }

    public static synchronized OrtSession.SessionOptions options() throws OrtException {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setIntraOpNumThreads(2);
        // idle worker threads would otherwise busy-wait between frames and burn CPU
        opts.addConfigEntry("session.intra_op.allow_spinning", "0");
        opts.addConfigEntry("session.inter_op.allow_spinning", "0");
        if (cuda == null) {
            cuda = !"cpu".equalsIgnoreCase(System.getenv("TVD_DEVICE"));
            if (cuda) {
                try {
                    new OrtSession.SessionOptions().addCUDA(0);
                    LOG.info("ONNX Runtime: dùng GPU (CUDA)");
                } catch (OrtException | UnsatisfiedLinkError e) {
                    cuda = false;
                    LOG.warning("ONNX Runtime: không dùng được CUDA (" + e.getMessage().lines().findFirst().orElse("")
                            + "); chạy trên CPU");
                }
            }
        }
        if (cuda) {
            // input sizes vary (letterboxed frames, vehicle crops for plates); an exhaustive cuDNN search on every
            // new shape costs far more than it saves
            ai.onnxruntime.providers.OrtCUDAProviderOptions cudaOpts = new ai.onnxruntime.providers.OrtCUDAProviderOptions(0);
            cudaOpts.add("cudnn_conv_algo_search", "HEURISTIC");
            opts.addCUDA(cudaOpts);
        }
        return opts;
    }
}
